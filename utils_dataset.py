# -*- coding: utf-8 -*-
import random
from functools import partial
from itertools import repeat
from typing import Callable
import xarray as xr
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from osgeo import gdal
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

import torch
import numpy as np
import joblib

def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data

# 读图像文件
def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影
    # print(dataset)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

# 读图像文件
def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影
    # print(dataset)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset



def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def fast_collate_for_prediction(batch):
    """ A fast collation function optimized for float32 images (np array or torch)
        and float32 targets (video prediction labels) in video prediction tasks"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into
        # one tensor ordered by position such that all tuple of position n will end up
        # in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.float32)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            # all input tensor tuples must be same length
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.zeros((batch_size, *batch[1][0].shape), dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class PrefetchLoader:
    """通用预加载器。

    - 支持 loader 每次返回任意个数的元素（数量由数据本身决定，可用 return_num 校验）
    - 任意位置的元素都可以是张量，或张量组成的 list/tuple（递归搬到 GPU）
    - 自动判断第一个元素是否为列表，结果保存在 first_is_list 属性中
    """

    def __init__(self, loader, return_num=None, fp16=False):
        self.loader = loader
        self.return_num = return_num   # 期望的返回元素个数，None 表示不校验
        self.fp16 = fp16
        self._first_is_list = None     # 迭代开始（或调用 check_first_is_list）后确定

    @staticmethod
    def _to_gpu(x, fp16):
        if isinstance(x, (list, tuple)):
            return [PrefetchLoader._to_gpu(i, fp16) for i in x]
        x = x.cuda(non_blocking=True)
        return x.half() if fp16 else x

    def __iter__(self):
        stream = torch.cuda.Stream()
        it = iter(self.loader)

        try:
            nxt = next(it)
        except StopIteration:
            return  # loader 为空时直接结束

        if self.return_num is not None:
            assert len(nxt) == self.return_num, \
                f"期望每次返回 {self.return_num} 个元素，实际得到 {len(nxt)} 个"

        self._first_is_list = isinstance(nxt[0], (list, tuple))

        first = True
        while True:
            with torch.cuda.stream(stream):
                moved = [self._to_gpu(t, self.fp16) for t in nxt]
            if not first:
                yield cur
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream)
            cur = moved
            try:
                nxt = next(it)
            except StopIteration:
                break
        yield cur

    def check_first_is_list(self):
        """在迭代开始前尽量判断第一个元素是否为列表；判断不了时返回 None。

        通过取 dataset[0] 来推断（适用于 map-style dataset + 默认 collate）。
        """
        if self._first_is_list is None:
            try:
                sample = self.loader.dataset[0]
                self._first_is_list = isinstance(sample[0], (list, tuple))
            except (TypeError, KeyError, IndexError, AttributeError):
                pass
        return self._first_is_list

    @property
    def first_is_list(self):
        """第一个元素是否为列表。迭代开始前若无法从 dataset 推断，则为 None。"""
        return self.check_first_is_list()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  num_workers=1,
                  num_aug_repeats=0,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=True,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=False,
                  worker_seeding='all',
                  return_num=None):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                print('setting the dis sampler')
        else:
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(loader, return_num=return_num, fp16=fp16)

    return loader, sampler


if __name__ == '__main__':
    pass