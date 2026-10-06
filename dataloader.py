# -*- coding: utf-8 -*-
import io
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import time
import pickle
from utils_dataset import create_loader
from datetime import datetime, timedelta

def generate_time_series(start, end, interval):
    current_time = start
    while current_time <= end:
        yield current_time
        current_time += timedelta(minutes=interval)

def list_files(directory):
    list_file = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_file.append(os.path.join(root, file))

    return list_file

def list_subdirectories(main_folder):
    subdirectories = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    return subdirectories

import numpy as np
import torch
from bisect import bisect_left, bisect_right

#

import os

import random

import numpy as np
from torch.utils.data import Dataset

import zarr



import numpy as np


class ERA5_Nexrad_mrms_qpe_usa_2km_dataset_test(Dataset):
    def __init__(self, file_list_L1, file_list_era5, file_list_qpe, file_list_rqi, full_series, time_embed, train_index_sample,
                 seg_len, lon_len=64, lat_len=32,
                 bs=32, in_len_val=5, sample_inte=4, mode='test', time_inte=None,
                 list_len=8, is_training=False, ch_num_13=4, idx_dim=None, product_name=None, seg_size=(512, 512), args=None):
        super(ERA5_Nexrad_mrms_qpe_usa_2km_dataset_test, self).__init__()
        import zarr
        print('len_train_loader', len(train_index_sample))
        # self.data = data
        self.product_name = product_name
        self.sample_inte = sample_inte
        self.in_len_val = in_len_val
        self.zarr_group_radar = zarr.open(file_list_L1[0], mode='r')
        self.radar_data = self.zarr_group_radar[file_list_L1[1]]
        self.zarr_group_era5 = zarr.open(file_list_era5[0], mode='r')
        self.era5_data = self.zarr_group_era5[file_list_era5[1]]
        self.zarr_group_qpe = zarr.open(file_list_qpe[0], mode='r')
        self.qpe_data = self.zarr_group_qpe[file_list_qpe[1]]
        self.zarr_group_rqi = zarr.open(file_list_rqi[0], mode='r')
        self.rqi_data = self.zarr_group_rqi[file_list_rqi[1]]
        self.full_time_coords = self.zarr_group_rqi['coords/full_time'][:]
        print('len(self.full_time_coords)', len(self.full_time_coords))
        print('self.rqi_data', self.rqi_data.shape)
        # print(self.full_time_coords[:10])
        # print(self.full_time_coords[-10:])

        self.args = args

        self.full_series = full_series
        self.time_embed = time_embed
        self.index_sample = train_index_sample
        self.bs, self.seg_len, self.lon_len, self.lat_len, self.seg_size = bs, seg_len, lon_len, lat_len, seg_size
        self.ch_num = 48
        self.time_inte = time_inte
        self.idx = 0
        self.border_tar = args.border_tar

        hr_shape = (args.L1_shape[-2], args.L1_shape[-1])
        # 低分辨率形状
        lr_shape = (args.era5_shape[-2], args.era5_shape[-1])
        self.sample_lat_size = hr_shape[0] - seg_size[0]# + 1
        self.sample_lon_size = hr_shape[1] - seg_size[1]# + 1

        # 计算缩放比例
        self.scale_i = hr_shape[0] / lr_shape[0]  # 1200 / 97 ≈ 12.371
        self.scale_j = hr_shape[1] / lr_shape[1]  # 2300 / 185 ≈ 12.432

        self.era5_seg_size = (int(self.seg_size[0] / self.scale_i)+1, int(self.seg_size[1] / self.scale_j)+1)

        # 步长：seg_size 的一半
        stride_h, stride_w = args.crop_stride_test[0], args.crop_stride_test[1]

        # 预计算所有滑窗起始索引
        self.test_indices = []
        self.test_indices_era5 = []
        # self.test_indices_dem = []
        for lat_idx in range(self.args.lat_dismiss[0], self.sample_lat_size-self.args.lat_dismiss[1], stride_h):
            for lon_idx in range(self.args.lon_dismiss[0], self.sample_lon_size-self.args.lon_dismiss[1], stride_w):
                self.test_indices.append((lat_idx, lon_idx))
                self.test_indices_era5.append(self.hr_to_lr_index(lat_idx, lon_idx))
                # self.test_indices_dem.append(self.lr_to_hr_index(lat_idx, lon_idx))
        # print(self.test_indices)
        # self.init_lat_lon()

    def init_lat_lon(self):
        # 边界范围
        lat_range = [49, 25]
        lon_range = [245, 291]
        # 生成等间距点
        lats = np.linspace(lat_range[0], lat_range[1], 1200)
        lons = np.linspace(lon_range[0], lon_range[1], 2300)
        # 归一化
        lats_norm = (lats - lat_range[0]) / (lat_range[1] - lat_range[0])
        lons_norm = (lons - lon_range[0]) / (lon_range[1] - lon_range[0])
        # 使用meshgrid创建网格
        lats_grid, lons_grid = np.meshgrid(lats_norm, lons_norm, indexing='ij')
        # indexing='ij' 使得输出形状为 (1200, 2300)
        # 拼接成最终结果
        self.geo_coords = np.concatenate([lats_grid[None, ...], lons_grid[None, ...]], axis=0)
        print(f"geo_coords shape: {self.geo_coords.shape}")  # 输出: (2, 1200, 2300)

    def hr_to_lr_index(self, hr_i, hr_j):
        """将高分辨率索引转换为低分辨率索引"""
        lr_i = int(hr_i / self.scale_i)  # lat
        lr_j = int(hr_j / self.scale_j)  # lon
        return (lr_i, lr_j)

    def __getitem__(self, idx):
        # idx = idx * self.sample_inte
        idx_sam = self.index_sample[idx]
        # rand_idx = (self.idx // self.bs) % len(self.time_inte)
        # rand_inte = self.time_inte[rand_idx]

        indices = slice(idx_sam, idx_sam + self.seg_len) # 40s one epoch
        if self.args.pure_test == 0:
            indices_in_len_val = slice(idx_sam, idx_sam + self.in_len_val)  # 40s one epoch  in_len_val
        else:
            indices_in_len_val = slice(idx_sam, idx_sam + self.in_len_val)  # 40s one epoch
        # radar_data_tmp = self.radar_data[indices_in_len_val]   # 尝试直接在这里用随机空间索引
        # era5_data_tmp = self.era5_data[indices_in_len_val]  # 尝试直接在这里用随机空间索引
        qpe_data_tmp = self.qpe_data[indices]  # 尝试直接在这里用随机空间索引
        # print(radar_data_tmp.shape)
        # print(era5_data_tmp.shape)
        # print(qpe_data_tmp.shape)
        # print(self.full_series[idx_sam+self.in_len_val-1])
        target_full = self.full_series[idx_sam+self.in_len_val].strftime("%m%d%H").encode('utf-8')
        # print(target_full)
        time_idx = np.where(self.full_time_coords == target_full)[0][0]
        # print(time_idx)
        indices_rqi = slice(time_idx, time_idx + self.seg_len - self.in_len_val)  # 40s one epoch
        rqi_data_tmp = self.rqi_data[indices_rqi]

        time_embedding = np.array(
            [self.time_embed[i] for i in range(idx_sam, idx_sam + self.seg_len)])

        # self.idx = self.idx + 1

        radar_data_out = np.array([self.radar_data[indices_in_len_val, :, lat_idx+self.border_tar[0]:lat_idx+self.seg_size[0]-self.border_tar[0],
                                   lon_idx+self.border_tar[1]:lon_idx+self.seg_size[1]-self.border_tar[1]] for (lat_idx, lon_idx) in self.test_indices])
        era5_data_out = np.array([self.era5_data[indices_in_len_val, :, lat_idx:lat_idx+self.era5_seg_size[0], lon_idx:lon_idx+self.era5_seg_size[1]] for (lat_idx, lon_idx) in self.test_indices_era5])
        if self.args.pure_test == 0:
            # if evaluate_map_figure, 注释rqi_data_tmp
            # rqi_data_tmp = rqi_data_tmp[:, None, ...]
            # rqi_data_tmp = np.array(
            #     [rqi_data_tmp[:, lat_idx + self.border_tar[0]:lat_idx + self.seg_size[0] - self.border_tar[0],
            #      lon_idx + self.border_tar[1]:lon_idx + self.seg_size[1] - self.border_tar[1]] for (lat_idx, lon_idx) in
            #      self.test_indices])
            qpe_data_out = np.array([qpe_data_tmp[:, :, lat_idx+self.border_tar[0]:lat_idx+self.seg_size[0]-self.border_tar[0],
                                     lon_idx+self.border_tar[1]:lon_idx+self.seg_size[1]-self.border_tar[1]] for (lat_idx, lon_idx) in self.test_indices])
        else:
            qpe_data_out = np.array([qpe_data_tmp[:self.in_len_val, :,
                                     lat_idx + self.border_tar[0]:lat_idx + self.seg_size[0] - self.border_tar[0],
                                     lon_idx + self.border_tar[1]:lon_idx + self.seg_size[1] - self.border_tar[1]] for
                                     (lat_idx, lon_idx) in self.test_indices])
            # rqi_data_out = np.array(
        # rqi_data_out = np.array(
        #     [rqi_data_tmp[:, lat_idx + self.border_tar[0]:lat_idx + self.seg_size[0] - self.border_tar[0],
        #      lon_idx + self.border_tar[1]:lon_idx + self.seg_size[1] - self.border_tar[1]] for (lat_idx, lon_idx) in
        #      self.test_indices])
        # coords_data_out = np.array([self.geo_coords[:, lat_idx+self.border_tar[0]:lat_idx + self.seg_size[0]-self.border_tar[0],
        #                             lon_idx+self.border_tar[1]:lon_idx + self.seg_size[1]-self.border_tar[1]] for (lat_idx, lon_idx) in self.test_indices])

        # print(radar_data_out.shape)
        # print(era5_data_out.shape)
        # print(qpe_data_out.shape)
        # print(coords_data_out.shape)
        # era5_idx = self.hr_to_lr_index(random_lat_idx)
        return qpe_data_tmp, radar_data_out, era5_data_out, qpe_data_out, np.transpose(time_embedding), rqi_data_tmp#, coords_data_out#, rand_idx

    def __len__(self):
        return len(self.index_sample) #// self.sample_inte


def load_ERA5_nexrad_mrms_qpe_2001_2017_usa_2km(batch_size, val_batch_size, test_batch_size, lon_len, lat_len,
                                                     L1_data_path, era5_data_path, mrms_qpe_path, rqi_path, file_path_dem, num_workers=4,
                           in_shape=[10, 1, 64, 64], distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
                           test=False, data_root_dir=None, args=None):
    # print(args.L1_shape)
    from datetime import datetime

    # hours = (datetime(2018, 1, 1) - datetime(2001, 1, 1)).days * 24
    # print(hours)
    # exit()
    file_name_L1, file_name_era5, file_name_qpe, file_name_rqi = 'Reflectivity', 'data', 'precipitation', 'data'
    time_embed = np.load(os.path.join(data_root_dir, 'radar/3D_NEXRAD/2016_2017_1hour/', "time_emb_2001_2030_cos_sin.npy"))[hours:]
    # print(time_embed.shape)
    # exit()
    # path = os.path.join(L1_data_path, "nexrad_2022/nexrad-2022-upload.hdf5")
    # data = h5py.File(path, 'r')
    image_size = in_shape[-1] if in_shape is not None else 64
    ch_num_13 = 8
    # seg_len = args.input_time_length + args.aft_seq_length_train
    start_time = datetime(2008, 1, 1, 0, 0, 0)  # *****************
    end_time = datetime(2021, 12, 31, 23, 0, 0)  # *****************
    interval = 60
    full_series = list(generate_time_series(start_time, end_time, interval))

    index_file = np.arange(len(full_series))
    # index_sample = [np.arange(len(full_series)) for _ in range(len(args.pred_len)+1)]
    index_sample = np.arange(len(full_series))

    if args.test:
        start_date0 = datetime(2008, 1, 1)
        end_date0 = datetime(2017, 1, 1, 0)
        # 计算时间差
        time_difference = end_date0 - start_date0
        # 将天数转换为小时
        # total_seconds() 返回总秒数，然后除以3600得到小时
        total_hours0 = int(time_difference.total_seconds() / 3600) - 1

    gap0, gap1 = 0, total_hours0
    start_index = (gap0 - args.sample_inte * max(args.time_inte) * (
                args.input_time_length + max(args.pred_len) - 1))  # 第一个索引
    start_index = start_index if start_index > 0 else 0  # 如果索引 <= 0，则设为 0
    index_file[start_index:(gap1 + 1)] = -1

    start_index2 = (gap0 - (args.sample_inte * (args.in_len_val + args.aft_seq_length_test - 1) + 1))  # 第2个索引
    start_index2 = start_index2 if start_index2 > 0 else 0  # 如果索引 <= 0，则设为 0
    index_sample[start_index2:(gap1 + args.sample_inte)] = -1  # *****************

    b = args.in_len_val + args.aft_seq_length_test  # 去掉后20%的最后b个索引
    c = args.in_len_val + args.aft_seq_length_val  # 去掉后20%的最后b个索引
    # import numpy as np

    # 生成连续的数字索引
    start_date = np.datetime64('2008-01-01T00:00')
    end_date = np.datetime64('2021-12-31T23:00')
    time_range = np.arange(start_date, end_date + np.timedelta64(60, 'm'), np.timedelta64(60, 'm'))

    # 生成数字索引（从1开始）
    numeric_indices = np.arange(0, len(time_range))

    # 按月划分索引
    months = {}
    for timestamp, index in zip(time_range, numeric_indices):
        month = timestamp.astype('datetime64[M]').astype(str)
        if month not in months:
            months[month] = []
        months[month].append(index)

    val_indices = {}
    test_indices = {}

    for month, indices in months.items():
        total_indices = len(indices)

        # 计算分割点
        train_split = int(total_indices * args.trainset_ratio)  # 80% 训练集分割点
        val_split = int(total_indices * (args.trainset_ratio + args.valset_ratio))  # 85% 验证集结束点

        # 验证集：接下来的5%
        val_part = indices[train_split:val_split]
        if b > 0 and len(val_part) > b:
            val_part = val_part[:-b]  # 去掉最后b个索引
        val_indices[month] = val_part

        # 测试集：最后15%
        test_part = indices[val_split:]
        if c > 0 and len(test_part) > c:
            test_part = test_part[:-c]  # 去掉最后c个索引
        test_indices[month] = test_part

    # 将结果合并为三个连续的数组
    val_all = np.concatenate(list(val_indices.values()))
    test_all = np.concatenate(list(test_indices.values()))

    val_last_20_hours = time_range[val_all]  # numeric_indices从1开始，需要-1
    val_last_20_hour_only = (
            (val_last_20_hours - val_last_20_hours.astype('datetime64[D]')) / np.timedelta64(1, 'h')).astype(int)
    # print(val_last_20_hour_only[:10])
    test_last_20_hours = time_range[test_all]  # numeric_indices从1开始，需要-1
    test_last_20_hour_only = (
            (test_last_20_hours - test_last_20_hours.astype('datetime64[D]')) / np.timedelta64(1, 'h')).astype(int)

    selected_hours_val = [i - args.in_len_val + 1 for i in range(24, 0, -args.sample_inte_val)]
    print('selected_hours_val', selected_hours_val)
    mask = np.isin(val_last_20_hour_only, selected_hours_val)
    val_last_20_all_filtered = val_all[mask]
    selected_hours = [i - args.in_len_val + 1 for i in range(24, 0, -args.sample_inte_test)]
    print('selected_hours', selected_hours)

    mask = np.isin(test_last_20_hour_only, selected_hours)
    test_last_20_all_filtered = test_all[mask]

    val_index_sample = index_sample[val_last_20_all_filtered]
    test_index_sample = index_sample[test_last_20_all_filtered]

    val_index_sample = [x for x in val_index_sample if x != -1]
    test_index_sample = [x for x in test_index_sample if x != -1]

    if test:
        # var_data_train, var_data_val, time_diff_emb_train, time_diff_emb_val = None, None, None, None
        dataloader_train, sampler_train = None, None

        val_set = ERA5_Nexrad_mrms_qpe_usa_2km_dataset_test((L1_data_path, file_name_L1),
                                                                          (era5_data_path, file_name_era5),
                                                                          (mrms_qpe_path, file_name_qpe), (rqi_path, file_name_rqi), full_series,
                                                            time_embed, val_index_sample,
                                         seg_len=args.in_len_val + args.aft_seq_length_val,
                                         lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                         in_len_val=args.in_len_val,
                                         sample_inte=args.sample_inte_test, mode='val', product_name=args.product_name,
                                                          seg_size=(args.tar_size[0], args.tar_size[1]), args=args)

        dataloader_vali, _ = create_loader(val_set,
                                           batch_size=val_batch_size,
                                           shuffle=False, is_training=False,
                                           pin_memory=True, drop_last=drop_last,
                                           num_workers=4,
                                           distributed=False, use_prefetcher=True, return_num=6)
        # val_set = ERA5_dataset(np.asarray(result_var_val), np.asarray(time_var_val))  # **********************  overlap_step
        test_set = ERA5_Nexrad_mrms_qpe_usa_2km_dataset_test((L1_data_path, file_name_L1),
                                                                          (era5_data_path, file_name_era5),
                                                                          (mrms_qpe_path, file_name_qpe), (rqi_path, file_name_rqi), full_series,
                                                             time_embed, test_index_sample,
                                   seg_len=args.in_len_val + args.aft_seq_length_test,
                                   lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                          in_len_val=args.in_len_val,
                                          sample_inte=args.sample_inte_test, mode='test', product_name=args.product_name,
                                                           seg_size=(args.tar_size[0], args.tar_size[1]), args=args)  # **********************  overlap_step
        # test_set = ERA5_dataset(np.asarray(result_var_test, dtype='float32'), np.asarray(time_var_val, dtype='float32'))  # **********************  overlap_step

        dataloader_test, _ = create_loader(test_set,
                                        batch_size=val_batch_size,
                                        shuffle=False, is_training=False,
                                        pin_memory=True, drop_last=drop_last,
                                        num_workers=4,
                                        distributed=False, use_prefetcher=True, return_num=6)  # set distributed=False to assign the value to the fuxi framework
        del val_set, test_set
    # print('load time: ',time.time() - time0)

    return dataloader_train, sampler_train, dataloader_vali, dataloader_test, #sampler_train


if __name__ == '__main__':
    pass
