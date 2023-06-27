import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

from torchvision.transforms import Normalize, Compose, Resize
from torchvision.transforms.functional import InterpolationMode

import timm

# Check the correct version of [timm] is installed.
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import ProcessTrainImage, TTensor
from models_counting_network import CountingNetwork
import open_clip


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Training Open-world Text-specified Object Counting Network"
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )

    parser.add_argument("--epochs", default=1000, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument(
        "--lr",
        type=float,
        default=6.25e-6,
        help="learning rate",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="epochs to warmup lr"
    )

    parser.add_argument(
        "--output_dir",
        default="./results",
        help="path where to save model and log",
    )

    parser.add_argument("--device", default="cuda", help="device to use for training")

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--resume",
        default="",
        help="file name for model checkpoint to resume from (leave empty to not use a checkpoint)",
    )

    parser.add_argument("--start_epoch", default=0, type=int)

    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_false",
        help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )

    parser.add_argument(
        "--img_dir",
        default="/scratch/local/hdd/nikian/images_384_VarV2",
        help="directory containing images from FSC-147",
    )

    parser.add_argument(
        "--gt_dir",
        default="/scratch/local/hdd/nikian/gt_density_map_adaptive_384_VarV2",
        help="directory containing ground truth binary dot annotation maps",
    )

    parser.add_argument(
        "--class_file",
        default="/scratch/local/hdd/nikian/ImageClasses_FSC147.txt",
        help="name of file with FSC-147 image class names",
    )

    parser.add_argument(
        "--FSC147_anno_file",
        default="/scratch/local/hdd/nikian/annotation_FSC147_384.json",
        help="name of file with FSC-147 annotations",
    )

    parser.add_argument(
        "--FSC147_D_anno_file",
        default="./FSC-147-D.json",
        help="name of file with FSC-147-D",
    )

    parser.add_argument(
        "--data_split_file",
        default="/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json",
        help="name of file with train, val, test splits of FSC-147",
    )

    return parser


# See https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/open_clip/transform.py
open_clip_vit_b_16_preprocess = Compose(
    [
        Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class TrainData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir
        self.gt_dir = args.gt_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split["train"]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.class_dict = {}
        with open(args.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                self.class_dict[key] = val

        self.transform_train = ProcessTrainImage(
            self.img_dir,
            self.fsc147_annotations,
            self.fsc147_d_annotations,
            self.class_dict,
            self.img,
        )

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = fsc147_d_anno["text_description"]

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        density_path = self.gt_dir + "/" + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype("float32")

        sample = {
            "image": image,
            "text": text,
            "gt_density": density,
            "dots": dots,
            "id": im_id,
        }
        sample = self.transform_train(sample)
        return (
            open_clip_vit_b_16_preprocess(sample["image"]),
            sample["gt_density"],
            sample["text"],
        )


class ValData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split["val"]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = self.clip_tokenizer(fsc147_d_anno["text_description"]).squeeze(-2)

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        W, H = image.size

        # This resizing step exists for consistency with CounTR's data resizing step.
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        image = Resize((new_H, new_W))(image)
        image = TTensor(image)

        return image, dots, text


def main(args):

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Fix a random seed, and force PyTorch to be deterministic for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    # NOTE: some operations during training do not have deterministic alternatives (such as [upsample_bilinear2d_backward_out_cuda]). Therefore, the line below is not executed.
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    dataset_train = TrainData(args)
    dataset_val = ValData(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Initialize the model.
    model = CountingNetwork()

    model.to(device)

    print("Model = %s" % str(model))

    print("lr: %.2e" % args.lr)

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    misc.load_model_FSC(args=args, model_without_ddp=model)

    print(f"Start training for {args.epochs} epochs")

    # Save the best MAE for the validation set.
    best_val_mae = math.inf
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)
        print_freq = 20

        train_mae = 0
        train_rmse = 0
        avg_loss = 0

        optimizer.zero_grad()

        for data_iter_step, (samples, gt_density, text_descriptions) in enumerate(
            metric_logger.log_every(data_loader_train, print_freq, header)
        ):

            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader_train) + epoch, args
            )

            samples = samples.to(device, non_blocking=True).half()
            gt_density = gt_density.to(device, non_blocking=True).half()
            text_descriptions = text_descriptions.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(samples, text_descriptions)

            # Compute the loss.
            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (output.shape[0], 1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(device)
            loss = (output - gt_density) ** 2
            loss = (loss * masks / (384 * 384)).sum() / output.shape[0]

            loss_value = loss.item()

            # Update information on the MAE and RMSE.
            batch_mae = 0
            batch_rmse = 0
            for i in range(output.shape[0]):
                pred_cnt = torch.sum(output[i] / 60).item()
                gt_cnt = torch.sum(gt_density[i] / 60).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                batch_mae += cnt_err
                batch_rmse += cnt_err**2

                if i == 0:
                    print(
                        f"{data_iter_step}/{len(data_loader_train)}: loss: {loss_value},  pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt}, AE: {cnt_err},  SE: {cnt_err ** 2}"
                    )

            train_mae += batch_mae
            train_rmse += batch_rmse
            avg_loss += loss_value

            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=True,
            )
            optimizer.zero_grad()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        curr_train_mae = train_mae / len(dataset_train)
        curr_train_rmse = (train_rmse / len(dataset_train)) ** 0.5
        avg_loss = avg_loss / len(data_loader_train)

        # Calculate the MAE and RMSE for the validation set for each epoch.
        val_mae = 0
        val_rmse = 0
        model.eval()
        for data_iter_step, (samples, gt_dots, text_description) in enumerate(
            iter(data_loader_val)
        ):

            samples = samples.to(device, non_blocking=True)
            gt_dots = gt_dots.to(device, non_blocking=True).half()
            text_description = text_description.to(device, non_blocking=True)

            _, _, h, w = samples.shape

            # Apply sliding window density map averaging technique used in CounTR.
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    (output,) = model(
                        open_clip_vit_b_16_preprocess(
                            samples[:, :, :, start : start + 384]
                        ),
                        text_description,
                    )
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0 : prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1 : 384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start : prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1 : w])

                    density_map = (
                        density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                    )

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

            gt_cnt = gt_dots.shape[1]
            cnt_err = abs(pred_cnt - gt_cnt)
            val_mae += cnt_err
            val_rmse += cnt_err**2

        curr_val_mae = val_mae / len(dataset_val)
        curr_val_rmse = (val_rmse / len(dataset_val)) ** 0.5

        # Save the model if it achieves the best MAE on the validation set.
        if curr_val_mae < best_val_mae:
            # Update the best MAE on the validation set and the epoch that achieved that MAE.
            best_val_mae = curr_val_mae
            # The model will be saved in the output directory with the file name "checkpoint-[args.epochs].pth".
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=args.epochs,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "Current Train MAE": curr_train_mae,
            "Current Train RMSE": curr_train_rmse,
            "Current Val MAE": curr_val_mae,
            "Current Val RMSE": curr_val_rmse,
            "epoch": epoch,
        }

        print(
            "Current Train MAE: {:5.2f}, Current Train RMSE: {:5.2f} Current Val MAE: {:5.2f}, Current Val RMSE: {:5.2f} ".format(
                curr_train_mae,
                curr_train_rmse,
                curr_val_mae,
                curr_val_rmse,
            )
        )

        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
