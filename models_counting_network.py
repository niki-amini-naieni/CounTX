from functools import partial
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed


class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Feature interaction module specifics.
        self.fim_num_img_tokens = img_encoder_num_output_tokens

        # Use a fixed sin-cos embedding.
        self.fim_pos_embed = nn.Parameter(
            torch.zeros(1, self.fim_num_img_tokens, fim_embed_dim), requires_grad=False
        )

        self.fim_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    fim_embed_dim,
                    fim_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(fim_depth)
            ]
        )

        self.fim_norm = norm_layer(fim_embed_dim)

        # --------------------------------------------------------------------------
        # Density map decoder regresssion module specifics.

        self.decode_head0 = nn.Sequential(
            nn.Conv2d(fim_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # CLIP model specifics (contains image and text encoder modules).

        self.clip_model = open_clip.create_model(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )

        # Freeze all the weights of the text encoder.
        vis_copy = copy.deepcopy(self.clip_model.visual)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.visual = vis_copy

        # Exemplar encoder with CNN
        self.decoder_proj1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[3,64,64]->[64,32,32]
        )
        self.decoder_proj2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[64,32,32]->[128,16,16]
        )
        self.decoder_proj3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # [128,16,16]->[256,8,8]
        )
        self.decoder_proj4 = nn.Sequential(
            nn.Conv2d(256, fim_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
            # [256,8,8]->[512,1,1]
        )

    def initialize_weights(self):
        # Initialize the positional embedding for the feature interaction module.
        fim_pos_embed = get_2d_sincos_pos_embed(
            self.fim_pos_embed.shape[-1],
            int(self.fim_num_img_tokens**0.5),
            cls_token=False,
        )
        self.fim_pos_embed.data.copy_(
            torch.from_numpy(fim_pos_embed).float().unsqueeze(0)
        )

        # Initialize nn.Linear and nn.LayerNorm layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use Xavier uniform weight initialization following the official JAX ViT.
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_img_encoder(self, imgs):
        return self.clip_model.encode_image(imgs)

    def forward_txt_encoder(self, counting_queries):
        return self.clip_model.encode_text(counting_queries)
    
    def forward_exemplar_encoder(self, exemplars):
        # Exemplar encoder
        shot_num = exemplars.shape[1]
        exemplars = exemplars.transpose(0, 1) # exemplars [N,3,3,64,64]->[3,N,3,64,64]
        y1 = []
        for yi in exemplars:
            yi = self.decoder_proj1(yi)
            yi = self.decoder_proj2(yi)
            yi = self.decoder_proj3(yi)
            yi = self.decoder_proj4(yi)
            N, C, _, _ = yi.shape
            y1.append(yi.squeeze(-1).squeeze(-1)) # yi [N,C,1,1]->[N,C]       
            
        y = torch.cat(y1,dim=0).reshape(shot_num,N,C)
        y = y.transpose(0,1) # y [3,N,C]->[N,3,C]
        return y

    def forward_fim(self, img_tokens, txt_tokens, exemplar_tokens):
        # Add positional embedding to image tokens.
        img_tokens = img_tokens + self.fim_pos_embed

        # Combine text and visual exemplar tokens.
        y = torch.cat([txt_tokens, exemplar_tokens], dim=1)

        # Pass image tokens and counting query tokens through the feature interaction module.
        x = img_tokens
        for blk in self.fim_blocks:
            x = blk(x, y)

        return self.fim_norm(x)

    def forward_decoder(self, fim_output_tokens):
        # Reshape the tokens output by the feature interaction module into a square feature map with [fim_embed_dim] channels.
        n, hw, c = fim_output_tokens.shape
        h = w = int(math.sqrt(hw))
        x = fim_output_tokens.transpose(1, 2).reshape(n, c, h, w)

        # Upsample output of this map to be N x [fim_embed_dim] x 24 x 24, as it was in CounTR.
        x = F.interpolate(x, size=24, mode="bilinear", align_corners=False)

        # Pass [x] through the density map regression decoder and upsample output until density map is the size of the input image.
        x = F.interpolate(
            self.decode_head0(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head1(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head2(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head3(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the channel dimension from [x], as the density map only has 1 channel.
        return x.squeeze(-3)

    def forward(self, imgs, counting_queries, exemplars):
        img_tokens = self.forward_img_encoder(imgs)
        # Add a token dimension to the CLIP text embeddings.
        txt_tokens = self.forward_txt_encoder(counting_queries).unsqueeze(-2)
        exemplar_tokens = self.forward_exemplar_encoder(exemplars)

        # Normalize tokens from all the modalities.
        img_tokens = F.normalize(img_tokens, dim=-1)
        txt_tokens = F.normalize(txt_tokens, dim=-1)
        exemplar_tokens = F.normalize(exemplar_tokens, dim=-1)

        fim_output_tokens = self.forward_fim(img_tokens, txt_tokens, exemplar_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred
