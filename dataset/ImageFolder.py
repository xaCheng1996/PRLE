import logging
import math
import random

import PIL.ImageChops
import cv2
import torch
import torch.utils.data as data
import json
import PIL.Image
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import os.path
import pandas as pd
import numpy as np
from heatmap_finetune import finetune_mask
from heatmap_fusion import show_pil_image
from .manipulation import *
from landmark_detect import *


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, mode='train'):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if mode == 'train':
                        if os.path.exists(os.path.dirname(path).replace('FF++_Faces_c40', 'heat_map_fusion')):
                            # print(os.path.dirname(path).replace('FF++_Faces_c40', 'FF++_faces_new'))
                            item = (path, class_to_idx[target])
                            images.append(item)
                    else:
                        item = (path, class_to_idx[target])
                        images.append(item)
                    # else:
                    #     print(path.replace('FF++_Faces_c40', 'FF++_faces_new'))
    return images


def make_ffpp_dataset_subset(dir, class_to_idx, extensions, subsets, mask_ratio=None):
    """
    path/train/fake/Deepfakes/video
    """
    images = []
    dir = os.path.expanduser(dir)
    logging.info("%s stage: use the subset %s" % (os.path.basename(dir), str(subsets)))
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        if target == 'real':
            if mask_ratio is None:
                d_sub = os.path.join(d, 'youtube')
            else:
                d_sub = os.path.join(d, 'youtube', 'per_%s' % str(mask_ratio))
            for root, _, fnames in sorted(os.walk(d_sub)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        elif target == 'fake':
            for subset_name in subsets:
                if mask_ratio is None:
                    d_sub = os.path.join(d, subset_name)
                else:
                    d_sub = os.path.join(d, subset_name, 'per_%s' % str(mask_ratio))
                for root, _, fnames in sorted(os.walk(d_sub)):
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, extensions):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
    return images




class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, mask_path=None, transform=None, target_transform=None, root_2=None,
                 config=None):
        classes, class_to_idx = find_classes(root)

        self.root = root
        self.dataset_mode = os.path.basename(self.root)

        # root = root.replace('train', 'test')
        # self.root = self.root.replace('train', 'test')
        self.dataset_mode_1 = os.path.basename(self.root)

        if config['data']['split'] and os.path.basename(os.path.dirname(root)) == 'FF++_faces':
            if self.dataset_mode == 'train':
                samples = make_ffpp_dataset_subset(root, class_to_idx, extensions, config['data']['dataset_name'])
            else:
                samples = make_ffpp_dataset_subset(root, class_to_idx, extensions, config['data']['dataset_name_val'])
        else:
            samples = make_dataset(root, class_to_idx, extensions, mode=self.dataset_mode)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions
        self.mask_path = mask_path

        self.config = config
        self.save_mask = self.config['general']['save_mask']
        self.use_pr_mask = self.config['data']['use_pr_mask']
        self.use_lmk_mask = self.config['data']['use_lmk_mask']
        if self.save_mask:
            logging.info('save masks and origin images with ration %s from root %s and mask path %s' %
                         (self.config['progress'][0], self.root,
                          os.path.join(
                              self.config['data']['save_img_path'], self.config['data']['dataset_name'][0],
                              self.dataset_mode, 'origin')))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if not self.config['data']['use_mask']:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.dataset_mode == 'train':
                sample = np.array(sample)
                # sample = self.add_noise(sample)
                sample = Image.fromarray(sample)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            if self.dataset_mode == 'train':
                path, target = self.samples[index]
                sample = self.loader(path)
                sample_return = np.array(sample.resize((256, 256)))
                m = None
                mask_image = None
                mask_image_show = []
                if self.mask_path is not None:
                    img_name = os.path.basename(path)
                    video_name = os.path.basename(os.path.dirname(path))
                    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    classes = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))

                    if self.use_pr_mask:
                        m_path = os.path.join(self.mask_path, self.dataset_mode_1, classes, dataset_name, video_name,
                                              "%s.npy" % img_name)
                        #
                        mask = np.load(m_path)
                        sample_resize = sample.resize(mask.shape[:2])
                        progress = [0.0 if random.randint(0, 100) < 50 else random.uniform(0.5, 1.0)]
                        if progress[0] != 0.0:
                            mask_finetune = finetune_mask(fusion_heatmap=mask, progress=progress)
                        else:
                            mask_finetune = {}
                            # if not os.path.isfile(os.path.join(save_m_path, "%s.npy" % img_name)):
                            #     np.save(os.path.join(save_m_path, "%s.npy" % img_name), mask_finetune[0])
                        m = mask_finetune
                    elif self.use_lmk_mask:
                        sample_resize = sample.resize((256, 256))
                        lmk_list = landmark_detect(sample_resize)
                        if len(lmk_list) == 0:
                            m_new = []
                        else:
                            rand_choice = random.randint(0, 3)  # 0 nose; 1 eyes; 2 mouth;
                            lmk = None
                            if rand_choice == 0:
                                lmk = lmk_list[27:36]
                                # point = get_convex_hull_coordinates(lmk)
                                # left, top, right, bottom = min(lmk[:][0]), min(lmk[:][1]), max(lmk[:][0]), max(lmk[:][1])
                                # for left_i in range(left, right):
                                #     for top_i in range(top, bottom):
                                #         m_new.append(((top_i, left_i), 1.0))
                            elif rand_choice == 1:
                                lmk = lmk_list[36:48]
                                # left, top, right, bottom = min(lmk[:][0]), min(lmk[:][1]), max(lmk[:][0]), max(lmk[:][1])
                                # for left_i in range(left, right):
                                #     for top_i in range(top, bottom):
                                #         m_new.append(((top_i, left_i), 1.0))
                            else:
                                lmk = lmk_list[48:]
                                # left, top, right, bottom = min(lmk[:][0]), min(lmk[:][1]), max(lmk[:][0]), max(lmk[:][1])
                                # for left_i in range(left, right):
                                #     for top_i in range(top, bottom):
                                #         m_new.append(((top_i, left_i), 1.0))
                            m_new = get_convex_hull_coordinates(lmk)
                        m = m_new

                    mask_image = self.add_mask(sample_resize, m)
                    mask_image = PIL.Image.fromarray(mask_image)
                    mask_image = self.transform(mask_image)

                    if self.save_mask:
                        heat_map_image_path = os.path.join(self.config['data']['save_img_path'],
                                                           dataset_name,
                                                           self.dataset_mode, 'heat', classes, video_name)
                        save_mask_image_path = os.path.join(
                            self.config['data']['save_img_path'], dataset_name,
                            self.dataset_mode, 'mask', classes, video_name)

                        # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                        # heatmap = Image.fromarray(heatmap)
                        sample_array = np.array(sample_resize)
                        result = show_cam_on_image(sample_array / 255, mask, use_rgb=True)
                        result = Image.fromarray(result)
                        # save_origin_image_path = os.path.join(
                        #     self.config['data']['save_img_path'], self.config['data']['dataset_name'],
                        #     self.dataset_mode, 'origin', 'per_%.1f' % self.config['progress'][0], classes, video_name)
                        os.makedirs(heat_map_image_path, exist_ok=True)
                        os.makedirs(save_mask_image_path, exist_ok=True)
                        # if not os.path.exists(save_origin_image_path):
                        #     os.makedirs(save_origin_image_path)
                        # mask_image_show = Image.fromarray(mask_image_show)
                        result.save(os.path.join(heat_map_image_path, img_name))
                        # mask_image_save.save(os.path.join(save_m_path, img_name))
                        # sample_resize.save(os.path.join(save_origin_image_path, img_name))
                    # from heatmap_fusion import show_pil_image
                    # show_pil_image([mask_image_save])
                # if self.transform is not None:
                #     sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return mask_image, target
            else:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target

    def add_mask(self, image, mask):
        image = np.array(image)
        new_image = image
        for k in mask:
            new_image[k[0][1]][k[0][0]][0] = random.randint(0, 256)
            new_image[k[0][1]][k[0][0]][1] = random.randint(0, 256)
            new_image[k[0][1]][k[0][0]][2] = random.randint(0, 256)
        return new_image

    def add_noise(self, img):
        # apply jpeg_encode augmentor

        # if rng.rand() > 0.7:
        #     b_img = jpeg_encode(img, quality=int(15 + rng.rand() * 65))
        #     img = jpeg_decode(b_img)
        # return img
        # do normalize first
        rng = np.random
        if rng.rand() > 0.5:
            img = (img - img.min()) / (img.max() - img.min() + 1) * 255

        # quantization noise
        if rng.rand() > .5:
            ih, iw = img.shape[:2]
            noise = rng.randn(ih // 4, iw // 4) * 2
            noise = cv2.resize(noise, (iw, ih))
            img = np.clip(img + noise[:, :, np.newaxis], 0, 255)

        # apply HSV augmentor
        if rng.rand() > 0.75:
            img = np.array(img, 'uint8')
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            if rng.rand() > 0.5:
                if rng.rand() > 0.5:
                    r = 1. - 0.5 * rng.rand()
                else:
                    r = 1. + 0.15 * rng.rand()
                hsv_img[:, :, 1] = np.array(
                    np.clip(hsv_img[:, :, 1] * r, 0, 255), 'uint8'
                )

            if rng.rand() > 0.5:
                # brightness
                if rng.rand() > 0.5:
                    r = 1. + rng.rand()
                else:
                    r = 1. - 0.5 * rng.rand()
                hsv_img[:, :, 2] = np.array(
                    np.clip(hsv_img[:, :, 2] * r, 0, 255), 'uint8'
                )

            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        img = self.adjust_gamma(img, (0.6 + rng.rand() * 0.8))

        if rng.rand() > 0.7:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 10) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), rng.randint(3))

        if rng.rand() > 0.7:
            if rng.rand() > 0.5:
                img = self.gaussian_noise(rng, img, rng.randint(15, 22))
            else:
                img = self.gaussian_noise(rng, img, rng.randint(0, 5))

        if rng.rand() > 0.7:
            # append color tone adjustment
            rand_color = tuple([60 + 195 * rng.rand() for _ in range(3)])
            img = self.adjust_tone(img, rand_color, rng.rand() * 0.3)

        # apply interpolation
        x, y = img.shape[:2]
        if rng.rand() > 0.75:
            r_ratio = rng.rand() + 1  # 1~2
            target_shape = (int(x / r_ratio), int(y / r_ratio))
            self.resize_rand_interp(rng, img, target_shape)
            self.resize_rand_interp(rng, img, (x, y))

        return np.array(img, 'uint8')

    def _clip_normalize(self, img):
        return np.clip(img, 0, 255).astype('uint8')

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        return self._clip_normalize(img + rng.randn(*img.shape) * sigma)

    def adjust_gamma(self, img, gamma):
        k = 1.0 / gamma
        img = cv2.exp(k * cv2.log(img.astype('float32') + 1e-15))
        f = 255.0 ** (1 - k)
        return self._clip_normalize(img * f)

    def get_linear_motion_kernel(self, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)

        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return None

        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)

        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s

        return kern

    def linear_motion_blur(self, img, angle, length):
        kern = self.get_linear_motion_kernel(angle, length)
        return cv2.filter2D(img, -1, kern)

    def adjust_tone(self, src, color, p):
        dst = (
                (1 - p) * src + p * np.ones_like(src) * np.array(color).reshape(
            (1, 1, len(color))))
        return self._clip_normalize(dst)

    def resize_rand_interp(self, rng, img, size):
        return cv2.resize(
            img, size, )

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = PIL.Image.open(f)
        except:
            print(path)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mask_path=None, transform=None, target_transform=None, config=None, root_2=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          mask_path=mask_path,
                                          transform=transform,
                                          config=config,
                                          target_transform=target_transform,
                                          root_2=root_2, )
        self.imgs = self.samples

