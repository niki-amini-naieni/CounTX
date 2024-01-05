import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import open_clip


class ProcessTrainImage(object):
    """
    **CounTR augmentation pipeline**. Refer to https://github.com/Verg-Avesta/CounTR/blob/54457c0fe2ce9962ac00fccee784570cbdc0a131/util/FSC147.py for further details.

    Resize the image so that:
        1. Its size is 384 x 384
        2. Its height and new width are divisible by 16
        3. Its aspect ratio is possibly preserved

    Density map is cropped to have the same size and horizontal offset as the cropped image.

    Augmentations including random crop, Gaussian noise, color jitter, Gaussian blur, random affine, random horizontal flip, and mosaic are used.
    """

    def __init__(
        self,
        img_dir,
        fsc147_annotations,
        fsc147_d_annotations,
        class_dict,
        train_set,
        max_hw=384,
    ):
        self.img_dir = img_dir
        self.fsc147_annotations = fsc147_annotations
        self.fsc147_d_annotations = fsc147_d_annotations
        self.class_dict = class_dict
        self.train_set = train_set
        self.max_hw = max_hw
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __call__(self, sample):
        image, bboxes, text, density, dots, im_id = (
            sample["image"],
            sample["bboxes"],
            sample["text"],
            sample["gt_density"],
            sample["dots"],
            sample["id"],
        )

        rects = list()
        cnt = 0
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            box = TF.crop(image, y1, x1, y2 + 1 - y1, x2 + 1 - x1)
            box = transforms.Resize((64, 64))(box)
            rects.append(box)
            cnt += 1
            if cnt == 3:
                break

        rects = torch.stack(rects)
        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor = float(new_W) / W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))

        # Random augmentation.
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if aug_p < 0.4:
            aug_flag = 1
            if aug_p < 0.25:
                aug_flag = 0
                mosaic_flag = 1

        resized_image = TTensor(resized_image)

        # Add Gaussian noise.
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Add color jitter and Gaussian blur.
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine
        if aug_flag == 1:
            re1_image = re_image.transpose(0, 1).transpose(1, 2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(
                    Keypoint(
                        x=min(new_W - 1, int(dots[i][0] * scale_factor)),
                        y=min(new_H - 1, int(dots[i][1])),
                    )
                )
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential(
                [
                    iaa.Affine(
                        rotate=(-15, 15),
                        scale=(0.8, 1.2),
                        shear=(-10, 10),
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    )
                ]
            )
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce the dot annotation map.
            resized_density = np.zeros(
                (resized_density.shape[0], resized_density.shape[1]), dtype="float32"
            )
            for i in range(len(kps.keypoints)):
                if (
                    int(kps_aug.keypoints[i].y) <= new_H - 1
                    and int(kps_aug.keypoints[i].x) <= new_W - 1
                ) and not kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][
                        int(kps_aug.keypoints[i].x)
                    ] = 1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Add a random horizontal flip.
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)

        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros(
                    (resized_density.shape[0], resized_density.shape[1]),
                    dtype="float32",
                )
                for i in range(dots.shape[0]):
                    resized_density[min(new_H - 1, int(dots[i][1]))][
                        min(new_W - 1, int(dots[i][0] * scale_factor))
                    ] = 1
                resized_density = torch.from_numpy(resized_density)

            # Take a random 384 x 384 crop.
            start = random.randint(0, new_W - 1 - 383)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            reresized_density = resized_density[:, start : start + 384]
        # Apply a random self mosaic using the same image or four different images.
        else:
            image_array = []
            map_array = []
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70:
                for i in range(4):
                    length = random.randint(150, 384)
                    start_W = random.randint(0, new_W - length)
                    start_H = random.randint(0, new_H - length)
                    reresized_image1 = TF.crop(
                        resized_image, start_H, start_W, length, length
                    )
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(
                        reresized_image1
                    )
                    reresized_density1 = np.zeros((resize_l, resize_l), dtype="float32")
                    for i in range(dots.shape[0]):
                        if (
                            min(new_H - 1, int(dots[i][1])) >= start_H
                            and min(new_H - 1, int(dots[i][1])) < start_H + length
                            and min(new_W - 1, int(dots[i][0] * scale_factor))
                            >= start_W
                            and min(new_W - 1, int(dots[i][0] * scale_factor))
                            < start_W + length
                        ):
                            reresized_density1[
                                min(
                                    resize_l - 1,
                                    int(
                                        (min(new_H - 1, int(dots[i][1])) - start_H)
                                        * resize_l
                                        / length
                                    ),
                                )
                            ][
                                min(
                                    resize_l - 1,
                                    int(
                                        (
                                            min(
                                                new_W - 1,
                                                int(dots[i][0] * scale_factor),
                                            )
                                            - start_W
                                        )
                                        * resize_l
                                        / length
                                    ),
                                )
                            ] = 1
                    reresized_density1 = torch.from_numpy(reresized_density1)
                    image_array.append(reresized_image1)
                    map_array.append(reresized_density1)
            else:
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0, 3)
                else:
                    gt_pos = random.randint(0, 4)
                for i in range(4):
                    if i == gt_pos:
                        Tim_id = im_id
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                        Tscale_factor = scale_factor
                    else:
                        Tim_id = self.train_set[
                            random.randint(0, len(self.train_set) - 1)
                        ]
                        Tdots = np.array(self.fsc147_annotations[Tim_id]["points"])
                        Timage = Image.open("{}/{}".format(self.img_dir, Tim_id))
                        Timage.load()
                        new_TH = 16 * int(Timage.size[1] / 16)
                        new_TW = 16 * int(Timage.size[0] / 16)
                        Tscale_factor = float(new_TW) / Timage.size[0]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                    length = random.randint(250, 384)
                    start_W = random.randint(0, new_TW - length)
                    start_H = random.randint(0, new_TH - length)
                    r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    r_density1 = np.zeros((resize_l, resize_l), dtype="float32")
                    if self.class_dict[im_id] == self.class_dict[Tim_id]:
                        for i in range(Tdots.shape[0]):
                            if (
                                min(new_TH - 1, int(Tdots[i][1])) >= start_H
                                and min(new_TH - 1, int(Tdots[i][1])) < start_H + length
                                and min(new_TW - 1, int(Tdots[i][0] * Tscale_factor))
                                >= start_W
                                and min(new_TW - 1, int(Tdots[i][0] * Tscale_factor))
                                < start_W + length
                            ):
                                r_density1[
                                    min(
                                        resize_l - 1,
                                        int(
                                            (
                                                min(new_TH - 1, int(Tdots[i][1]))
                                                - start_H
                                            )
                                            * resize_l
                                            / length
                                        ),
                                    )
                                ][
                                    min(
                                        resize_l - 1,
                                        int(
                                            (
                                                min(
                                                    new_TW - 1,
                                                    int(Tdots[i][0] * Tscale_factor),
                                                )
                                                - start_W
                                            )
                                            * resize_l
                                            / length
                                        ),
                                    )
                                ] = 1
                    r_density1 = torch.from_numpy(r_density1)
                    image_array.append(r_image1)
                    map_array.append(r_density1)

            reresized_image5 = torch.cat(
                (
                    image_array[0][:, blending_l : resize_l - blending_l],
                    image_array[1][:, blending_l : resize_l - blending_l],
                ),
                1,
            )
            reresized_density5 = torch.cat(
                (
                    map_array[0][blending_l : resize_l - blending_l],
                    map_array[1][blending_l : resize_l - blending_l],
                ),
                0,
            )
            for i in range(blending_l):
                reresized_image5[:, 192 + i] = image_array[0][
                    :, resize_l - 1 - blending_l + i
                ] * (blending_l - i) / (2 * blending_l) + reresized_image5[
                    :, 192 + i
                ] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
                reresized_image5[:, 191 - i] = image_array[1][:, blending_l - i] * (
                    blending_l - i
                ) / (2 * blending_l) + reresized_image5[:, 191 - i] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)

            reresized_image6 = torch.cat(
                (
                    image_array[2][:, blending_l : resize_l - blending_l],
                    image_array[3][:, blending_l : resize_l - blending_l],
                ),
                1,
            )
            reresized_density6 = torch.cat(
                (
                    map_array[2][blending_l : resize_l - blending_l],
                    map_array[3][blending_l : resize_l - blending_l],
                ),
                0,
            )
            for i in range(blending_l):
                reresized_image6[:, 192 + i] = image_array[2][
                    :, resize_l - 1 - blending_l + i
                ] * (blending_l - i) / (2 * blending_l) + reresized_image6[
                    :, 192 + i
                ] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
                reresized_image6[:, 191 - i] = image_array[3][:, blending_l - i] * (
                    blending_l - i
                ) / (2 * blending_l) + reresized_image6[:, 191 - i] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)

            reresized_image = torch.cat(
                (
                    reresized_image5[:, :, blending_l : resize_l - blending_l],
                    reresized_image6[:, :, blending_l : resize_l - blending_l],
                ),
                2,
            )
            reresized_density = torch.cat(
                (
                    reresized_density5[:, blending_l : resize_l - blending_l],
                    reresized_density6[:, blending_l : resize_l - blending_l],
                ),
                1,
            )
            for i in range(blending_l):
                reresized_image[:, :, 192 + i] = reresized_image5[
                    :, :, resize_l - 1 - blending_l + i
                ] * (blending_l - i) / (2 * blending_l) + reresized_image[
                    :, :, 192 + i
                ] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
                reresized_image[:, :, 191 - i] = reresized_image6[
                    :, :, blending_l - i
                ] * (blending_l - i) / (2 * blending_l) + reresized_image[
                    :, :, 191 - i
                ] * (
                    i + blending_l
                ) / (
                    2 * blending_l
                )
            reresized_image = torch.clamp(reresized_image, 0, 1)

        # Apply a Gaussian filter to the density map.
        reresized_density = ndimage.gaussian_filter(
            reresized_density.numpy(), sigma=(1, 1), order=0
        )

        # Scale up the density map by a factor of 60.
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)

        # Tokenize the text description.
        text = self.clip_tokenizer(text).squeeze(-2)

        # Return the processed sample.
        sample = {
            "image": reresized_image,
            "text": text,
            "gt_density": reresized_density,
            "exemplars": rects
        }

        return sample


TTensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

Augmentation = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
        ),
        transforms.GaussianBlur(kernel_size=(7, 9)),
    ]
)
