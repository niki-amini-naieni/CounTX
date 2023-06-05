# model.py
from functools import partial
import math
import numpy as np
import copy

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import open_clip

from vit import CrossAttentionBlock
from pos_embed import get_2d_sincos_pos_embed


class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Feature interaction module (fim) specifics.
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
                for i in range(fim_depth)
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

    def foward_txt_encoder(self, counting_queries, shot_num):
        # Exchange batch and shot dimensions.
        counting_queries = counting_queries.transpose(0, 1)  # [N, S, 77]->[S, N, 77]
        out_lst = []
        cnt = 0
        for counting_query_shot in counting_queries:
            cnt += 1
            if cnt > shot_num:
                break
            counting_query_tokens = self.clip_model.encode_text(counting_query_shot)
            N, C = counting_query_tokens.shape
            out_lst.append(counting_query_tokens)

        counting_query_tokens = torch.cat(out_lst, dim=0).reshape(shot_num, N, C)

        # Return batch and shot dimensions to their original order.
        counting_query_tokens = counting_query_tokens.transpose(
            0, 1
        )  # [S, N, C]->[N, S, C]

        return counting_query_tokens

    def forward_fim(self, img_tokens, txt_tokens):
        # Add positional embedding to image tokens.
        img_tokens = img_tokens + self.fim_pos_embed

        # Pass image tokens and counting query tokens through the feature interaction module.
        x = img_tokens
        for blk in self.fim_blocks:
            x = blk(x, txt_tokens)

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

    def forward(self, imgs, counting_queries, shot_num):
        img_tokens = self.forward_img_encoder(imgs)
        txt_tokens = self.foward_txt_encoder(counting_queries, shot_num)
        fim_output_tokens = self.forward_fim(img_tokens, txt_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred


def main_counting_network(**kwargs):
    model = CountingNetwork(
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


"""
Worked Example:

# Model and image file names.
model_file_name = "./model.pth"
image_file_name = "./espresso_spoons.jpg"

# Load model.
model = main_counting_network()
checkpoint = torch.load(model_file_name, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)

# Define preprocessing.
tokenizer = open_clip.get_tokenizer("ViT-B-16")

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

# Load and process the image.
image = Image.open(image_file_name)
image.load()
image = preprocess(image)

# Specify the text description,
# (i.e., response to "what object should be counted?").
text = "the gilt silver espresso spoons"

# Tokenize the text.
text = tokenizer(text)

# The shot number is 1, since there is only 1 text description.
shot_num = 1

# Place the model in eval mode.
model.eval()

# Infer the count.
with torch.no_grad():
    density_map = model(image.unsqueeze(0), text.unsqueeze(0), shot_num)
pred_cnt = torch.sum(density_map / 60).item()

print("Predicted Count: " + str(pred_cnt))
"""
