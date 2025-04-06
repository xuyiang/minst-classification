# coding:utf8
import gzip
import os
import torch as t
import numpy as np
from torch.utils import data
import torchvision.transforms as T

from config import opt


class Minst(data.Dataset):

    def __init__(self, data_root, label_root=None, transforms=None, train=True, test=False):
        """
        获取所有图片，并根据训练、验证、测试划分数据
        :param data_root:数据路径
        :param transforms:数据转换操作
        :param train:是否训练集
        :param test:是否测试集
        """
        # Check if files exist
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data file not found: {data_root}")
        if not test and not os.path.exists(label_root):
            raise FileNotFoundError(f"Label file not found: {label_root}")

        self.test = test
        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

        if test:
            self.image_nums = opt.test_image_nums
        else:
            self.image_nums = opt.train_image_nums

        # Load image data
        try:
            print(f"Loading data from {data_root}")
            with gzip.open(data_root, 'rb') as bytestream:
                magic = int.from_bytes(bytestream.read(4), 'big')
                if magic != 2051:
                    raise ValueError(f"Invalid magic number {magic} in MNIST image file {data_root}")
                num_images = int.from_bytes(bytestream.read(4), 'big')
                rows = int.from_bytes(bytestream.read(4), 'big')
                cols = int.from_bytes(bytestream.read(4), 'big')

                if num_images != self.image_nums:
                    raise ValueError(f"Expected {self.image_nums} images, got {num_images}")
                if rows != opt.image_size or cols != opt.image_size:
                    raise ValueError(f"Expected {opt.image_size}x{opt.image_size} images, got {rows}x{cols}")

                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                data = (data - (opt.pixel_depth / 2.0)) / opt.pixel_depth
                self.data = data.reshape(num_images, rows, cols)
                print(f"Loaded {num_images} images of size {rows}x{cols}")

        except Exception as e:
            raise RuntimeError(f"Error loading image data from {data_root}: {e}")

        # Load label data
        if not test:
            try:
                print(f"Loading labels from {label_root}")
                with gzip.open(label_root, 'rb') as bytestream:
                    magic = int.from_bytes(bytestream.read(4), 'big')
                    if magic != 2049:
                        raise ValueError(f"Invalid magic number {magic} in MNIST label file {label_root}")
                    num_items = int.from_bytes(bytestream.read(4), 'big')
                    if num_items != self.image_nums:
                        raise ValueError(f"Expected {self.image_nums} labels, got {num_items}")

                    buf = bytestream.read(num_items)
                    self.labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
                    print(f"Loaded {num_items} labels")

            except Exception as e:
                raise RuntimeError(f"Error loading label data from {label_root}: {e}")

        # Split into train/val
        if test:
            pass
        elif train:
            self.image_nums = int(opt.train_image_nums * 0.7)
            self.data = self.data[:self.image_nums]
            self.labels = self.labels[:self.image_nums]
            print(f"Using {self.image_nums} images for training")
        else:
            self.image_nums = int(opt.train_image_nums - opt.train_image_nums * 0.7)
            self.data = self.data[int(opt.train_image_nums * 0.7):]
            self.labels = self.labels[int(opt.train_image_nums * 0.7):]
            print(f"Using {self.image_nums} images for validation")

    def __getitem__(self, index):
        """
        返回一张图片的数据，如果是测试集没有label
        :param index:
        :return:
        """
        try:
            if not 0 <= index < len(self):
                raise IndexError(f"Index {index} out of range [0, {len(self)})")

            img = self.data[index]
            img = img.reshape(opt.image_size, opt.image_size).astype(np.float32)
            img = self.transforms(img)

            if self.test:
                return img, index
            else:
                return img, self.labels[index]

        except Exception as e:
            print(f"Error loading item {index}: {e}")
            # Return a zero tensor and label if there's an error
            if self.test:
                return t.zeros(1, opt.image_size, opt.image_size), index
            else:
                return t.zeros(1, opt.image_size, opt.image_size), 0

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_data = Minst(data_root="train/train-images.gz", label_root="train/train-labels.gz", train=True)
    train_dataloader = data.DataLoader(train_data, opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
    import torchvision.transforms as T

    for ii, (data, target) in enumerate(train_dataloader):
        toimage = T.ToPILImage()
        image = data[ii].numpy()
        image = (image + 0.5) * 255
        result = toimage(T.ToTensor()(image))
        result.show()
        print(target[ii])
