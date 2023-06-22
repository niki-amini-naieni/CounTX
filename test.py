import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import util.misc as misc
from util.FSC147 import TTensor
from models_counting_network import CountingNetwork
import open_clip


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Testing Open-world Text-specified Object Counting Network"
    )

    parser.add_argument(
        "--data_split",
        default="val",
        help="data split of FSC-147 to test",
    )

    parser.add_argument(
        "--output_dir",
        default="./test",
        help="path where to save test log",
    )

    parser.add_argument("--device", default="cuda", help="device to use for testing")

    parser.add_argument(
        "--resume",
        default="./counting_network.pth",
        help="file name for model checkpoint to use for testing",
    )

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


open_clip_vit_b_16_preprocess = transforms.Compose(
    [
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


class TestData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split[args.data_split]

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
        image = transforms.Resize((new_H, new_W))(image)
        image = TTensor(image)

        return image, dots, text


def main(args):

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Force PyTorch to be deterministic for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html.
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    dataset_test = TestData(args)
    print(dataset_test)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Initialize the model.
    model = CountingNetwork()

    model.to(device)

    misc.load_model_FSC(args=args, model_without_ddp=model)

    print(f"Start testing.")
    start_time = time.time()

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Testing (" + args.data_split + ")"
    print_freq = 20

    test_mae = 0
    test_rmse = 0

    for data_iter_step, (samples, gt_dots, text_descriptions) in enumerate(
        metric_logger.log_every(data_loader_test, print_freq, header)
    ):

        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        text_descriptions = text_descriptions.to(device, non_blocking=True)

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
                    text_descriptions,
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
        test_mae += cnt_err
        test_rmse += cnt_err**2

        print(
            f"{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  AE: {cnt_err},  SE: {cnt_err ** 2} "
        )

    print("Averaged stats:", metric_logger)

    log_stats = {
        "MAE": test_mae / (len(data_loader_test)),
        "RMSE": (test_rmse / (len(data_loader_test))) ** 0.5,
    }

    print(
        "Test MAE: {:5.2f}, Test RMSE: {:5.2f} ".format(
            test_mae / (len(data_loader_test)),
            (test_rmse / (len(data_loader_test))) ** 0.5,
        )
    )

    with open(
        os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
