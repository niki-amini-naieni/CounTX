import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import hub

import util.misc as misc
from models_reproduce_paper import main_counting_network
import open_clip

clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
TTensor = transforms.Compose([transforms.ToTensor()])


def get_args_parser():
    parser = argparse.ArgumentParser("Testing CARPK")

    parser.add_argument(
        "--output_dir", default="./test-carpk", help="path where to save the log"
    )

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--resume", default="carpk.pth", help="model to test")

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_false",
        help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )

    return parser


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    cudnn.benchmark = True

    dataset_test = hub.load("hub://activeloop/carpk-test")
    print(dataset_test)

    data_loader_test = dataset_test.pytorch(
        num_workers=args.num_workers, batch_size=1, shuffle=False
    )

    # Initialize the model.
    model = main_counting_network()

    model.to(device)

    misc.load_model_FSC(args=args, model_without_ddp=model)

    print(f"Start testing.")
    start_time = time.time()

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Testing CARPK"
    print_freq = 20

    test_mae = 0
    test_rmse = 0

    for data_iter_step, data in enumerate(
        metric_logger.log_every(data_loader_test, print_freq, header)
    ):

        samples = (data["images"] / 255).to(device, non_blocking=True)
        labels = data["labels"].to(device, non_blocking=True)
        samples = samples.transpose(2, 3).transpose(1, 2)
        new_h = 384
        new_w = 683

        samples = transforms.Resize((new_h, new_w))(samples)

        density_map = torch.zeros([new_h, new_w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < new_w:
                (output,) = model(
                    open_clip_vit_b_16_preprocess(
                        samples[:, :, :, start : start + 384]
                    ),
                    clip_tokenizer(["the cars"])
                    .unsqueeze(0)
                    .to(device, non_blocking=True),
                    1,
                )
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, new_w - prev - 1, 0, 0))
                d1 = b1(output[:, 0 : prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, new_w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1 : 384])

                b3 = nn.ZeroPad2d(padding=(0, new_w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start : prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1 : new_w])

                density_map = (
                    density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                )

                prev = start + 383
                start = start + 128
                if start + 383 >= new_w:
                    if start == new_w - 384 + 128:
                        break
                    else:
                        start = new_w - 384

        pred_cnt = torch.sum(density_map / 60).item()

        gt_cnt = labels.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        test_mae += cnt_err
        test_rmse += cnt_err**2

        print(
            f"{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2} "
        )

        torch.cuda.synchronize()

    print("Averaged stats:", metric_logger)

    log_stats = {
        "MAE": test_mae / (len(data_loader_test)),
        "RMSE": (test_rmse / (len(data_loader_test))) ** 0.5,
    }

    print(
        "Current MAE: {:5.2f}, RMSE: {:5.2f} ".format(
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
