import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from functools import partial
from ddm.utils import exists, convert_image_to_fn, normalize_to_neg_one_to_one
from PIL import Image, ImageDraw
import torch.nn.functional as F
import math
import torchvision.transforms.functional as F2
import torchvision.datasets as datasets
from typing import Any, Callable, Optional, Tuple
import os
import pickle
import numpy as np
import copy
# import albumentations
import random


class CIFAR10(datasets.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            img_folder: str,
            image_size=[32, 32],
            augment_horizontal_flip=False,
            with_class=False,
            target_transform: Optional[Callable] = None,
            normalize_to_neg_one_to_one=True, **kwargs
    ) -> None:

        super(CIFAR10, self).__init__(img_folder,
                                      target_transform=target_transform)
        self.data_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            # ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]
        self.data: Any = []
        self.targets = []
        self.with_class = with_class
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        # now load the picked numpy arrays
        for file_name, checksum in self.data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomCrop(image_size),
            T.ToTensor()
        ])
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.with_class:
            return {'image':img, 'class': target}
        else:
            return {'image': img}

    def __len__(self) -> int:
        return len(self.data)


class ImageDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        image_size,
        exts = ['jpg', 'png'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True, **kwargs
        ):
        super().__init__()
        self.img_folder = img_folder
        self.image_size = image_size
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'*.{ext}')]
        print('data_len:', len(self.img_paths))
        # self.mask_paths = [(Path(self.mask_folder) / f'{item.stem}_mask.jpg') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            # T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path)
        img = self.transform(img).to(torch.float32)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
        return {'image':img}

class ImageNetDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        image_size,
        exts = ['JPEG'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        random_crop=True, **kwargs
        ):
        super().__init__()
        self.img_folder = img_folder
        self.image_size = image_size
        self.random_crop = random_crop
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').rglob(f'*.{ext}')]
        ignore = "n06596364_9591.JPEG"
        self.img_paths = [ipath for ipath in self.img_paths if ignore not in str(ipath)]
        print('There are total {} images.'.format(len(self.img_paths)))
        # self.mask_paths = [(Path(self.mask_folder) / f'{item.stem}_mask.jpg') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            # T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomCrop(image_size),
            T.ToTensor()
        ])
        assert self.image_size[0] == self.image_size[1]
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.image_size[0])
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.image_size[0], width=self.image_size[1])
        else:
            self.cropper = albumentations.RandomCrop(height=self.image_size[0], width=self.image_size[1])
        self.flipper = albumentations.HorizontalFlip()
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.flipper])
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = self.preprocessor(image=img)["image"]
        # mask_path = self.mask_paths[index]
        # mask = Image.open(mask_path).convert('L')
        # img = self.transform(img)
        if self.normalize_to_neg_one_to_one:
            # img = normalize_to_neg_one_to_one(img)
            img = (img / 127.5 - 1.0).astype(np.float32)
        else:
            img = (img / 255).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return {'image':img}

class LSUNDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        image_size,
        exts = ['jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True, **kwargs
        ):
        super().__init__()
        self.img_folder = img_folder
        self.image_size = image_size
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'*.{ext}')]
        # self.mask_paths = [(Path(self.mask_folder) / f'{item.stem}_mask.jpg') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            # T.Lambda(maybe_convert_fn),
            # T.Resize(image_size, interpolation=3),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # default to score-sde preprocessing
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        img = Image.fromarray(img)

        # mask_path = self.mask_paths[index]
        # mask = Image.open(mask_path).convert('L')
        img = self.transform(img).to(torch.float32)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
        return {'image':img}

class ImageMaskDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        mask_folder,
        image_size,
        exts = ['jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None, **kwargs
    ):
        super().__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'*.{ext}')]
        self.mask_paths = [(Path(self.mask_folder) / f'{item.stem}_mask.jpg') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = Compose([
            Lambda(maybe_convert_fn),
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            RandomCrop(image_size),
            ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert('L')
        img, mask = self.transform(img, mask)
        img = normalize_to_neg_one_to_one(img)
        return {'image':img, 'cond': mask}

class InpaintDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        # mask_folder,
        image_size,
        exts = ['jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split = 'train',
        hole_range=[0, 1], **kwargs
    ):
        super().__init__()
        assert split in["train", "test"]
        self.img_folder = img_folder
        # self.mask_folder = mask_folder
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = sorted([p for ext in exts for p in Path(f'{img_folder}').glob(f'*.{ext}')])
        if split == "train":
            self.img_paths = self.img_paths[:-2000]
        else:
            self.img_paths = self.img_paths[-2000:]
        # self.mask_paths = [(Path(self.mask_folder) / f'{item.stem}_mask.jpg') for item in self.img_paths]
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.hole_range = hole_range
        if split == 'train':
            self.transform = Compose([
                # Lambda(maybe_convert_fn),
                # Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                # RandomCrop(image_size),
                T.ToTensor()
            ])
        else:
            self.transform = Compose([
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = img_path.name
        img = Image.open(img_path).convert('RGB')
        # mask_path = self.mask_paths[index]
        # mask = Image.open(mask_path).convert('L')
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        mask = self.random_mask(h, hole_range=self.hole_range)
        mask = torch.from_numpy(mask)
        img = Image.fromarray(img)
        img = self.transform(img).to(torch.float32)
        mask_img = mask * img
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
            mask_img = normalize_to_neg_one_to_one(mask_img)
        return {'image':img, 'cond': mask_img, 'ori_mask': mask, 'img_name': img_name}
    def random_mask(self, s, hole_range=[0,1]):
        coef = min(hole_range[0] + hole_range[1], 1.0)
        while True:
            mask = np.ones((s, s), np.uint8)
            def Fill(max_size):
                w, h = np.random.randint(max_size), np.random.randint(max_size)
                ww, hh = w // 2, h // 2
                x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
                mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
            def MultiFill(max_tries, max_size):
                for _ in range(np.random.randint(max_tries)):
                    Fill(max_size)
            MultiFill(int(4 * coef), s // 2)
            MultiFill(int(2 * coef), s)
            mask = np.logical_and(mask, 1 - RandomBrush(int(8 * coef), s))  # hole denoted as 0, reserved as 1
            hole_ratio = 1 - np.mean(mask)
            if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
                continue
            return mask[np.newaxis, ...].astype(np.float32)

def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

class CityscapesDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        exts = ['png'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train', **kwargs
    ):
        super().__init__()
        self.img_folder = Path(os.path.join(data_root, 'leftImg8bit', split))
        self.mask_folder = Path(os.path.join(data_root, 'gtFine', split))
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in self.img_folder.rglob(f'*.{ext}')]
        self.mask_paths = [(self.mask_folder / item.parent.stem / f'{item.stem[:-12]}_gtFine_labelTrainIds.png')
                           for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            # Lambda(maybe_convert_fn),
            Resize(image_size, interpolation=3, interpolation2=0),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # RandomCrop(image_size),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = Image.fromarray(img)
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype(np.uint8)
        mask += 1
        mask = Image.fromarray(mask)
        img, mask = self.transform(img, mask)
        img = F2.to_tensor(img)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
        mask = torch.from_numpy(np.array(mask)).to(torch.float32) / 19
        mask = mask.unsqueeze(0)
        # mask = mask.permute(2, 0, 1).contiguous()
        return {'image':img, 'cond': mask}

class ADE20KDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        exts = ['jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='training', **kwargs
    ):
        super().__init__()
        self.img_folder = Path(os.path.join(data_root, 'images', split))
        self.mask_folder = Path(os.path.join(data_root, 'annotations', split))
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in self.img_folder.rglob(f'*.{ext}')]
        self.mask_paths = [(self.mask_folder / f'{item.stem}.png') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            # Lambda(maybe_convert_fn),
            Resize(image_size, interpolation=3, interpolation2=0),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # RandomCrop(image_size),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # default to score-sde preprocessing
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        img = Image.fromarray(img)

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert('L')
        # default to score-sde preprocessing
        mask = np.array(mask).astype(np.uint8)
        mask = mask[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        assert mask.max() < 151 and mask.min() > -1;
        # mask[mask==255] = 0
        # print(mask.max())
        mask = Image.fromarray(mask)
        img, mask = self.transform(img, mask)
        img = F2.to_tensor(img)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
        mask = torch.from_numpy(np.array(mask)).to(torch.float32) / 150
        mask = mask.unsqueeze(0)
        # mask = mask.permute(2, 0, 1).contiguous()
        return {'image':img, 'cond': mask}

class SRDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        inter_type='bicubic',
        down=4, **kwargs
    ):
        super().__init__()
        self.img_folder = Path(img_folder)
        self.inter_type = inter_type
        self.down = down
        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[inter_type]
        # self.img_folder = Path(os.path.join(data_root, f'DIV2K_{split}_HR'))
        # self.mask_folder = Path(os.path.join(data_root, f'DIV2K_{split}_LR_{inter_type}_X{down}',
        #                                      f'DIV2K_{split}_LR_{inter_type}', f'X{down}'))
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in self.img_folder.rglob(f'*.{ext}')]
        # self.mask_paths = [(self.mask_folder / f'{item.stem}x{down}.png') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.random_crop = T.RandomCrop(size=image_size)
        if split == 'train':
            self.transform = Compose([
                # Lambda(maybe_convert_fn),
                # Resize(image_size, interpolation=3, interpolation2=0),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                # RandomCrop(image_size),
                ToTensor()
            ])
        else:
            self.transform = Compose([
                ToTensor()
            ])
        self.split = split

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # default to score-sde preprocessing
        img = self.random_crop(img)
        mask = copy.deepcopy(img)
        mask = mask.resize((self.image_size[0]//self.down, self.image_size[1]//self.down), resample=self.interpolation)
        img, mask = self.transform(img, mask)
        # img = F2.to_tensor(img)
        # mask = F2.to_tensor(mask)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
            mask = normalize_to_neg_one_to_one(mask)
        return {'image':img, 'cond': mask}

class SRDatasetTest(data.Dataset):
    def __init__(
        self,
        img_folder,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        inter_type='bicubic',
        down=4, **kwargs
    ):
        super().__init__()
        self.img_folder = Path(img_folder)
        self.inter_type = inter_type
        self.down = down
        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[inter_type]
        # self.img_folder = Path(os.path.join(data_root, f'DIV2K_{split}_HR'))
        # self.mask_folder = Path(os.path.join(data_root, f'DIV2K_{split}_LR_{inter_type}_X{down}',
        #                                      f'DIV2K_{split}_LR_{inter_type}', f'X{down}'))
        self.image_size = image_size
        # self.img_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        # self.mask_paths = [p for ext in exts for p in Path(f'{img_folder}').glob(f'**/*.{ext}')]
        self.img_paths = [p for ext in exts for p in self.img_folder.rglob(f'*.{ext}')]
        # self.mask_paths = [(self.mask_folder / f'{item.stem}x{down}.png') for item in self.img_paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.random_crop = T.RandomCrop(size=image_size)

        self.transform = Compose([
            ToTensor()
        ])
        self.split = split

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = img_path.name
        img = Image.open(img_path).convert('RGB')
        # default to score-sde preprocessing
        w, h = img.size
        new_width = math.ceil(w / 256) * 256
        new_height = math.ceil(h / 256) * 256
        res = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
        res.paste(img, (0, 0))
        # img = res
        mask = copy.deepcopy(res)
        mask = mask.resize((new_width//self.down, new_height//self.down), resample=self.interpolation)
        img, mask = self.transform(img, mask)
        # img = F2.to_tensor(img)
        # mask = F2.to_tensor(mask)
        if self.normalize_to_neg_one_to_one:
            img = normalize_to_neg_one_to_one(img)
            mask = normalize_to_neg_one_to_one(mask)
        return {'image': img, 'cond': mask, 'ori_size': (h, w), 'img_name': img_name}

class EdgeDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        # inter_type='bicubic',
        # down=4,
        threshold=0.3, use_uncertainty=False, **kwargs
    ):
        super().__init__()
        # self.img_folder = Path(img_folder)
        self.edge_folder = Path(os.path.join(data_root))
        self.img_folder = Path(os.path.join(data_root, f'imgs'))
        self.image_size = image_size

        self.edge_paths = [p for ext in exts for p in self.edge_folder.rglob(f'*.{ext}')]
        # self.img_paths = [(self.img_folder / f'{item.stem}.jpg') for item in self.edge_paths]
        self.threshold = threshold * 256
        self.use_uncertainty = use_uncertainty
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        # self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        # self.random_crop = RandomCrop(size=image_size)
        # self.transform = Compose([
        #     # Lambda(maybe_convert_fn),
        #     # Resize(image_size, interpolation=3, interpolation2=0),
        #     Resize(image_size, interpolation=InterpolationMode.BILINEAR, interpolation2=InterpolationMode.NEAREST),
        #     RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
        #     # RandomCrop(image_size),
        #     ToTensor()
        # ])

        self.transform = T.Compose([
            # Resize(self.image_size, interpolation=InterpolationMode.BILINEAR, interpolation2=InterpolationMode.NEAREST),
            T.ToTensor()])
        self.transform2 = T.Compose([
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.edge_paths)

    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        width, height = img.size
        width = int(width / 32) * 32
        height = int(height / 32) * 32
        img = img.resize((width, height), Image.Resampling.BILINEAR)
        # print("img.size:", img.size)
        img = self.transform(img)
        return img

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path).convert('L')
        lb = np.array(lb_data).astype(np.float32)
        # img = np.array(img).astype(np.uint8)
        # width, height = lb_data.size
        # width = int(width / 32) * 32
        # height = int(height / 32) * 32
        # lb_data = lb_data.resize((width, height), Image.Resampling.BILINEAR)
        # print("lb_data.size:", lb_data.size)
        # lb = np.array(lb_data, dtype=np.float32)
        # if lb.ndim == 3:
        #     lb = np.squeeze(lb[:, :, 0])
        # assert lb.ndim == 2
        threshold = self.threshold
        # lb = lb[np.newaxis, :, :]
        # lb[lb == 0] = 0

        # ---------- important ----------
        # if self.use_uncertainty:
        #     lb[np.logical_and(lb > 0, lb < threshold)] = 2
        # else:
        # lb[np.logical_and(lb > 0, lb < threshold)] /= 255.

        lb[lb >= threshold] = 255
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb

    def __getitem__(self, index):
        edge_path = self.edge_paths[index]
        # img_path = self.img_paths[index]

        # img = self.read_img(img_path)
        edge = self.read_lb(edge_path)
        edge = self.transform2(edge)

        # print("-------hhhhhhhhhhhhh--------:", img.shape, edge.shape)
        # edge = Image.open(edge_path).convert('L')
        # # default to score-sde preprocessing
        # mask = Image.open(img_path).convert('RGB')
        # edge, img = self.transform(edge, mask)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            edge = normalize_to_neg_one_to_one(edge)
            # img = normalize_to_neg_one_to_one(img)
        return {'image': edge}

class NYUDv2DepthDataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            **kwargs
    ):
        super().__init__()
        if split not in["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split))
        self.split = split
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*.jpg')]
        self.depth_images = []
        for p in self.rgb_images:
            p_name = p.name
            p2_name = p_name.replace('rgb_', 'sync_depth_')
            p2_name = p2_name.replace('.jpg', '.png')
            p2 = p.parent / p2_name
            self.depth_images.append(p2)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            RandomCrop(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        rgb_path = self.rgb_images[index]
        filename = rgb_path.name
        depth_path = self.depth_images[index]
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        depth = depth.crop((41, 45, 601, 471))
        rgb = rgb.crop((41, 45, 601, 471))
        if self.split == "train":
            rgb, depth = self.transform(rgb, depth)
        rgb = F2.to_tensor(rgb)
        depth = np.array(depth).astype(np.float32)
        # depth = np.clip(depth, 10, 10000)
        depth = torch.from_numpy(depth) / 10000
        depth = depth.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
            depth = normalize_to_neg_one_to_one(depth)
        return {'image': depth, 'cond': rgb, 'img_name': filename}

class NYUDv2DepthDataset2(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            **kwargs
    ):
        super().__init__()
        if split not in["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split))
        self.split = split
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*.jpg')]
        self.depth_images = []
        for p in self.rgb_images:
            p_name = p.name
            p2_name = p_name.replace('rgb_', 'sync_depth_')
            p2_name = p2_name.replace('.jpg', '.png')
            p2 = p.parent / p2_name
            self.depth_images.append(p2)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        rgb_path = self.rgb_images[index]
        filename = rgb_path.name
        depth_path = self.depth_images[index]
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        depth = depth.crop((41, 45, 601, 471))
        rgb = rgb.crop((41, 45, 601, 471))
        if self.split == "train":
            rgb, depth = self.transform(rgb, depth)
        elif self.split == "test":
            rgb, depth = self.transform_test(rgb, depth)
        else:
            raise NotImplementedError
        rgb = F2.to_tensor(rgb)
        depth = np.array(depth).astype(np.float32)
        # depth = np.clip(depth, 10, 10000)
        depth = torch.from_numpy(depth) / 10000
        depth = depth.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
            depth = normalize_to_neg_one_to_one(depth)
        return {'image': depth, 'cond': rgb, 'img_name': filename}

class DUTSDataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'DUTS-TR',
            'test': 'DUTS-TE',
        }
        self.split = split
        if split not in["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split]))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*.jpg')]
        self.gt_images = []
        for p in self.rgb_images:
            p_name = p.name
            p2_name = p_name.replace('.jpg', '.png')
            p2 = p.parent.parent / p.parent.name.replace('Image', 'Mask') / p2_name
            self.gt_images.append(p2)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                gt_path = self.gt_images[index]
                rgb = Image.open(rgb_path).convert("RGB")
                size = rgb.size
                gt = Image.open(gt_path).convert("L")
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        # if min(size) < self.image_size[0]:
        #     s = self.image_size[0] / min(size)
        #     new_size = (int(size[0]*s), int(size[1]*s))
        #     rgb = rgb.resize(size=new_size)
        #     gt = gt.resize(size=new_size)
        if self.split == 'train':
            rgb, gt = self.transform(rgb, gt)
        elif self.split == 'test':
            rgb, gt = self.transform_test(rgb, gt)
        else:
            raise NotImplementedError
        rgb = F2.to_tensor(rgb)
        gt = np.array(gt).astype(np.float32) / 255.
        gt = torch.from_numpy(gt)
        gt = gt.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
            gt = normalize_to_neg_one_to_one(gt)
        return {'image': gt, 'cond': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class SketchDataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'val',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, 'GT', split_map[split]))
            # self.sketcch_root = Path(os.path.join(data_root, 'Sketch', split_map[split]))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*.png') if not p.name.startswith('._')]
        self.sketch_images = []
        for p in self.rgb_images:
            p_name = p.name
            # p2_name = p_name.replace('.jpg', '.png')
            p2 = p.parent.parent.parent.parent / p.parent.parent.parent.name.replace('GT', 'Sketch') \
                 / p.parent.parent.name / p.parent.name / p_name
            self.sketch_images.append(p2)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                sketch_path = self.sketch_images[index]
                rgb = Image.open(rgb_path).convert("RGB")
                size = rgb.size
                sketch = Image.open(sketch_path).convert("L")
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        # if min(size) < self.image_size[0]:
        #     s = self.image_size[0] / min(size)
        #     new_size = (int(size[0]*s), int(size[1]*s))
        #     rgb = rgb.resize(size=new_size)
        #     gt = gt.resize(size=new_size)
        if self.split == 'train':
            rgb, sketch = self.transform(rgb, sketch)
        elif self.split == 'test':
            rgb, sketch = self.transform_test(rgb, sketch)
        else:
            raise NotImplementedError
        rgb = F2.to_tensor(rgb)
        sketch = np.array(sketch).astype(np.float32) / 255.
        sketch = torch.from_numpy(1. - sketch)
        sketch = sketch.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
            sketch = normalize_to_neg_one_to_one(sketch)
        return {'image': rgb, 'cond': sketch, 'img_name': filename, 'ori_size': (size[1], size[0])}
    

class SketchOnlyDataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'val',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, 'Sketch', split_map[split]))
            # self.sketcch_root = Path(os.path.join(data_root, 'Sketch', split_map[split]))
        self.image_size = image_size
        self.sketch_images = [p for p in self.data_folder.rglob(f'*.png') if not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.sketch_images)

    def __getitem__(self, index):
        while True:
            try:
                sketch_path = self.sketch_images[index]
                filename = sketch_path.name
                sktech = Image.open(sketch_path).convert("L")
                size = sktech.size
                break
            except:
                index = random.choice(range(len(self.sketch_images)))
        if self.split == 'train':
            sktech = self.transform(sktech)
        elif self.split == 'test':
            sktech = self.transform_test(sktech)
        else:
            raise NotImplementedError
        sktech = np.array(sktech).astype(np.float32) / 255.
        sktech = torch.from_numpy(1.-sktech)
        sktech = sktech.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            sktech = normalize_to_neg_one_to_one(sktech)
        return {'image': sktech, 'img_name': filename, 'ori_size': (size[1], size[0])}
    
class Paired_dataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='cat',
            target_folder_name='dog',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = sorted([p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')])
        self.B_images = sorted([p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')])
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomCrop((256, 256)),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index]
                B_path = self.B_images[index]
                # print(f"{A_path}, \n{B_path}")
                # B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path).convert("RGB")
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, target=B_img)
            # B_img = self.transform(B_img)
        elif self.split == 'test':
            A_img = self.transform_test(A_img)
            B_img = self.transform_test(B_img)
        else:
            raise NotImplementedError
        A_img = F2.to_tensor(A_img)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}

class RGB_Unpair_dataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='cat',
            target_folder_name='dog',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomCrop((256, 256)),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path).convert("RGB")
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = F2.to_tensor(A_img)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}

class Modality_Unpair_dataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='edges',
            target_folder_name='shoes',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path).convert("L")
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = np.array(A_img).astype(np.float32) / 255.
        A_img = torch.from_numpy(1.-A_img)
        A_img = A_img.unsqueeze(0)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}

class Modality_Unpair_dataset_sem(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='edges',
            target_folder_name='shoes',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path).convert("L")
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = torch.from_numpy(np.array(A_img).astype(np.float32) / 255.)
        A_img = A_img.unsqueeze(0)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}

class Modality_Unpair_dataset_helen_sem(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='edges',
            target_folder_name='shoes',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path).convert("L")
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        A_img = np.array(A_img).astype(np.uint8)
        crop = min(A_img.shape[0], A_img.shape[1])
        h, w, = A_img.shape[0], A_img.shape[1]
        A_img = A_img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        A_img = Image.fromarray(A_img)
        B_img = np.array(B_img).astype(np.uint8)
        crop = min(B_img.shape[0], B_img.shape[1])
        h, w, = B_img.shape[0], B_img.shape[1]
        B_img = B_img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        B_img = Image.fromarray(B_img)
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = torch.from_numpy(np.array(A_img).astype(np.float32) / 10.)
        A_img = A_img.unsqueeze(0)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}

class Modality_Unpair_dataset_depth(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='edges',
            target_folder_name='shoes',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path)
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        A_img = A_img.crop((41, 45, 601, 471))
        B_img = B_img.crop((41, 45, 601, 471))
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = torch.from_numpy(np.array(A_img).astype(np.float32) / 10000)
        A_img = A_img.unsqueeze(0)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}
        
class Modality_Unpair_dataset_kitti_depth(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            source_folder_name='edges',
            target_folder_name='shoes',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.A_path = Path(os.path.join(data_root, split_map[split], source_folder_name))
            self.B_path = Path(os.path.join(data_root, split_map[split], target_folder_name))
        self.crop_image_size = [320, 640]
        self.image_size = image_size
        self.A_images = [p for p in self.A_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.B_images = [p for p in self.B_path.rglob('*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.random_crop = T.RandomCrop(size=self.crop_image_size)
        self.transform = Compose([
            #CenterCrop(self.crop_image_size),
            Resize(image_size),
            # Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            # CenterCrop(self.crop_image_size),
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        while True:
            try:
                A_path = self.A_images[index % self.A_size]
                # B_path = self.B_images[index]
                B_path = self.B_images[random.randint(0, self.B_size - 1)]
                A_name = A_path.name
                B_name = B_path.name
                A_img = Image.open(A_path)
                B_img = Image.open(B_path).convert("RGB")
                break
            except:
                index = random.choice(range(len(self.A_images)))
        #A_img = self.random_crop(A_img)
        #B_img = self.random_crop(B_img)
        if self.split == 'train':
            A_img, B_img = self.transform(A_img, B_img)
        elif self.split == 'test':
            A_img, B_img = self.transform_test(A_img, B_img)
        else:
            raise NotImplementedError
        A_img = torch.from_numpy(np.array(A_img).astype(np.float32) / 65535)
        A_img = A_img.unsqueeze(0)
        B_img = F2.to_tensor(B_img)
        if self.normalize_to_neg_one_to_one:
            A_img = normalize_to_neg_one_to_one(A_img)
            B_img = normalize_to_neg_one_to_one(B_img)
        return {'src_img': A_img, 'trg_img': B_img, 'src_name': A_name, 'trg_name': B_name}
    
class Single_dataset_edge(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path).convert("L")
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = np.array(rgb).astype(np.float32) / 255.
        rgb = torch.from_numpy(1.-rgb)
        rgb = rgb.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}


class Single_dataset_celeba_mask(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path)
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.)
        rgb = rgb.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Single_dataset_sem(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path).convert("L")
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 10.)
        rgb = rgb.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Single_dataset_depth(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path)
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        rgb = rgb.crop((41, 45, 601, 471))
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 10000)
        rgb = rgb.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Single_dataset_kitti_depth(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.crop_image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.random_crop = T.RandomCrop(size=self.crop_image_size)
        self.transform = Compose([
            CenterCrop(self.crop_image_size),
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            CenterCrop(self.crop_image_size),
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path)
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        if self.split == 'train':
            # rgb = self.random_crop(rgb)
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 65535)
        rgb = rgb.unsqueeze(0)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Single_dataset_kitti_rgb(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.crop_image_size = image_size
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.random_crop = T.RandomCrop(size=self.crop_image_size)
        self.transform = Compose([
            CenterCrop(self.crop_image_size),
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            CenterCrop(self.crop_image_size),
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path).convert("RGB")
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        # rgb = self.random_crop(rgb)
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = F2.to_tensor(rgb)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Single_dataset(data.Dataset):
    def __init__(
            self,
            data_root,
            image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=True,
            split='train',
            datafolder_name='cat',
            **kwargs
    ):
        super().__init__()
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        self.split = split
        if split not in ["train", "test"]:
            self.data_folder = Path(data_root)
        else:
            self.data_folder = Path(os.path.join(data_root, split_map[split], datafolder_name))
        self.image_size = image_size
        self.rgb_images = [p for p in self.data_folder.rglob(f'*') if p.name.endswith(('.png', '.jpg')) and not p.name.startswith('._')]

        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.transform = Compose([
            Resize(image_size),
            T.RandomCrop((256, 256)),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])
        self.transform_test = Compose([
            Resize(image_size),
            # RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            # ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        while True:
            try:
                rgb_path = self.rgb_images[index]
                filename = rgb_path.name
                rgb = Image.open(rgb_path).convert("RGB")
                size = rgb.size
                break
            except:
                index = random.choice(range(len(self.rgb_images)))
        if self.split == 'train':
            rgb = self.transform(rgb)
        elif self.split == 'test':
            rgb = self.transform_test(rgb)
        else:
            raise NotImplementedError
        rgb = F2.to_tensor(rgb)
        if self.normalize_to_neg_one_to_one:
            rgb = normalize_to_neg_one_to_one(rgb)
        return {'image': rgb, 'img_name': filename, 'ori_size': (size[1], size[0])}

class Identity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        return input, target

class Resize(T.Resize):
    def __init__(self, size, interpolation2=None, **kwargs):
        super().__init__(size, **kwargs)
        if interpolation2 is None:
            self.interpolation2 = self.interpolation
        else:
            self.interpolation2 = interpolation2

    def forward(self, img, target=None):
        if target is None:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            return img
        else:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            target = F2.resize(target, self.size, self.interpolation2, self.max_size, self.antialias)
            return img, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target=None):
        if target is None:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
            return img
        else:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
                target = F2.hflip(target)
            return img, target

class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, target=None):
        if target is None:
            img = F2.center_crop(img, self.size)
            return img
        else:
            img = F2.center_crop(img, self.size)
            target = F2.center_crop(target, self.size)
            return img, target

class RandomCrop(T.RandomCrop):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    def single_forward(self, img, i, j, h, w):
        if self.padding is not None:
            img = F2.pad(img, self.padding, self.fill, self.padding_mode)
        width, height = F2.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F2.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F2.pad(img, padding, self.fill, self.padding_mode)

        return F2.crop(img, i, j, h, w)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.size)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        else:
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w)
            return img, target

class ToTensor(T.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, img, target=None):
        if target is None:
            img = F2.to_tensor(img)
            return img
        else:
            img = F2.to_tensor(img)
            target = F2.to_tensor(target)
            return img, target

class Lambda(T.Lambda):
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, img, target=None):
        if target is None:
            return self.lambd(img)
        else:
            return self.lambd(img), self.lambd(target)

class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, target=None):
        if target is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target


if __name__ == '__main__':
    dataset = CIFAR10(
        img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/cifar-10-python',
        augment_horizontal_flip=False
    )
    # dataset = CityscapesDataset(
    #     # img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
    #     data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/Cityscapes/',
    #     # data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/ADEChallengeData2016/',
    #     image_size=[512, 1024],
    #     exts = ['png'],
    #     augment_horizontal_flip = False,
    #     convert_image_to = None,
    #     normalize_to_neg_one_to_one=True,
    #     )
    # dataset = SRDataset(
    #     img_folder='/media/huang/ZX3 512G/data/DIV2K/DIV2K_train_HR',
    #     image_size=[512, 512],
    # )
    # dataset = InpaintDataset(
    #     img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
    #     image_size=[256, 256],
    #     augment_horizontal_flip = True
    # )
    dataset = EdgeDataset(
        data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/BSDS_my_aug96',
        image_size=[320, 320],
    )
    for i in range(len(dataset)):
        d = dataset[i]
        mask = d['cond']
        print(mask.max())
    dl = data.DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)

    pause = 0
