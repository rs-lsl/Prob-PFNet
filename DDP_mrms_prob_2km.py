import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from tqdm import tqdm
import sys
import pandas as pd
import gc
import tempfile
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import pickle
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils0 import compute_crps_fully_vectorized
from utils_dataset import write_img_gdal
import heapq
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import shutil
import logging
from utils_dataset import create_loader, read_img
from tqdm import tqdm
import math
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import uniform_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_mrms_qpe.log"),  # 保存到文件
        logging.StreamHandler()           # 终端显示
    ]
)



def std_nan_safe(tensor, dim=None, correction=1, keepdim=False):
    """
    安全的忽略NaN的标准差计算

    Args:
        tensor: 输入张量
        dim: 计算的维度
        correction: 贝塞尔校正 (0表示总体标准差，1表示样本标准差)
    """
    mask = ~torch.isnan(tensor)

    if dim is None:
        valid_data = tensor[mask]
        return torch.std(valid_data, correction=correction)
    else:
        valid_mean = torch.nanmean(tensor, dim=dim, keepdim=True)
        squared_diff = (tensor - valid_mean) ** 2
        squared_diff_clean = squared_diff.where(mask, torch.tensor(0.0, device=tensor.device))

        valid_count = mask.sum(dim=dim)
        variance = squared_diff_clean.sum(dim=dim) / (valid_count - correction)
        # print(torch.sqrt(variance).shape)
        return torch.sqrt(variance)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        raise EnvironmentError("NOT using distributed mode")
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dis_backend = 'nccl'
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def cleanup():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def is_main_process():
    return get_rank() == 0


def clip_grads(params, args, norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    args.clip_mode = args.clip_mode if args.clip_grad is not None else None
    if args.clip_mode is None:
        return
    if args.clip_mode == 'norm':
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad, norm_type=norm_type)
    elif args.clip_mode == 'value':
        torch.nn.utils.clip_grad_value_(params, args.clip_grad)
    else:
        assert False, f"Unknown clip mode ({args.clip_mode})."


class Pred_model(nn.Module):
    def __init__(self, model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test,
                 const_data, file_name_dem,
                 in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True,
                 time_emb_num=10, results_dir='', device=None, rank=0,
                 local_rank=0, loss_type='', cp_dir=None, args=None, **kwargs):
        super(Pred_model, self).__init__()
        self.args = args
        self.results_dir = results_dir
        self.cp_dir = cp_dir
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        # self.loss_weight = loss_weight  #
        B, T, C, H, W = in_shape  # T is input_time_length
        self.shape_val = [H, W]
        self.bs = B
        self.ch = C
        # self.target_dim = args.target_dim # list(range(args.p_dim, args.other_dim))+[15]  # ****************************

        self.dataloader_train, self.sampler_train, self.dataloader_val, self.dataloader_test = \
            dataloader_train, sampler_train, dataloader_val, dataloader_test
        # self.sampler_train = sampler_train
        self.const_data = None  # const_data.type(torch.float32).to(self.device, non_blocking=True)

        self.model = model

        if rank == 0:
            print("Total number of paramerters in networks is {}  ".format(
                sum(x.numel() for x in self.model.parameters())))

        log_path = os.path.join(self.results_dir, 'logs', args.ex_name)
        # self.logwriter = LogWriter(logdir=log_path)

        if not args.test:
            # self.steps_per_epoch = len(dataloader_train)
            self.init_optim(optimizer)

        self.init_lat_weight()
        self.init_max_indices()

        self.rain_list = []
        self.norain_list = []

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.results_dir, 'tensorboard', self.args.ex_name)
        )


        self.mean_accumulator = None
        self.total_samples = 0

        """
        Args:
            precip_thresholds: 降水强度阈值列表 (mm/h)
            lead_times: 预测时刻数
        """
        self.precip_thresholds = [torch.tensor(i) for i in args.threholds]
        self.precip_thresholds_6hour_accum = [torch.tensor(i*args.forecast_inte) for i in args.threholds]
        self.lead_times = args.aft_seq_length_test
        # 存储最优概率阈值: shape [lead_times, len(precip_thresholds)]
        self.optimal_prob_thresholds = None

        # self._buf2 = torch.empty_like(tensor)
        # self._buf2 = torch.empty_like(tensor)
        # print(file_name_dem)
        self.dem_data = read_img(file_name_dem)
        self.dem_data = torch.from_numpy(
            (self.dem_data - np.min(self.dem_data)) / (np.max(self.dem_data) - np.min(self.dem_data))).type(
            torch.float32).to(self.device)[None, ...]
        # print('self.dem_data.shape', self.dem_data.shape)

        hr_shape = (args.L1_shape[-2], args.L1_shape[-1])
        self.scale_up0, self.scale_up1 = args.dem_shape[0] / hr_shape[0], args.dem_shape[1] / hr_shape[1]
        self.dem_seg_size = (int(args.tar_L1_shape[-2] * self.scale_up0), int(args.tar_L1_shape[-1] * self.scale_up1))

        self.coords_seg_size = (args.tar_size[0], args.tar_size[1])
        self.init_lat_lon()

        stride_h, stride_w = args.crop_stride_test[0], args.crop_stride_test[1]
        self.test_indices = []
        self.test_indices_dem = []
        for lat_idx in range(self.args.lat_dismiss[0], self.sample_lat_size-self.args.lat_dismiss[1], stride_h):
            for lon_idx in range(self.args.lon_dismiss[0], self.sample_lon_size-self.args.lon_dismiss[1], stride_w):
                self.test_indices.append((lat_idx, lon_idx))
                self.test_indices_dem.append(self.lr_to_hr_index(lat_idx, lon_idx))

    def lr_to_hr_index(self, hr_i, hr_j):
        """将高分辨率索引转换为低分辨率索引"""
        lr_i = int(hr_i * self.scale_up0)  # lat
        lr_j = int(hr_j * self.scale_up1)  # lon
        return lr_i, lr_j

    def init_lat_lon(self):
        # 边界范围
        lat_range = [49, 25]
        lon_range = [245, 291]
        # 生成等间距点
        lats = np.linspace(lat_range[0], lat_range[1], 1201)
        lons = np.linspace(lon_range[0], lon_range[1], 2301)
        # 归一化
        lats_norm = (lats - lat_range[0]) / (lat_range[1] - lat_range[0])
        lons_norm = (lons - lon_range[0]) / (lon_range[1] - lon_range[0])
        # 使用meshgrid创建网格
        lats_grid, lons_grid = np.meshgrid(lats_norm, lons_norm, indexing='ij')
        # indexing='ij' 使得输出形状为 (1200, 2300)
        # 拼接成最终结果
        self.geo_coords = torch.from_numpy(np.concatenate([lats_grid[None, ...], lons_grid[None, ...]], axis=0)).type(
            torch.float16).to(self.device)
        print(f"geo_coords shape: {self.geo_coords.shape}")  # 输出: (2, 1200, 2300)

    def init_max_indices(self):
        hr_shape = (self.args.L1_shape[-2], self.args.L1_shape[-1])
        self.sample_lat_size_dis = hr_shape[0] - self.args.tar_size[0] - self.args.lat_dismiss[0] - self.args.lat_dismiss[1]
        self.sample_lon_size_dis = hr_shape[1] - self.args.tar_size[1] - self.args.lon_dismiss[0] - self.args.lon_dismiss[1]

        self.sample_lat_size = hr_shape[0] - self.args.tar_size[0]
        self.sample_lon_size = hr_shape[1] - self.args.tar_size[1]

        stride_h, stride_w = self.args.crop_stride_test[0], self.args.crop_stride_test[1]

        # 直接计算最后一个滑窗起始索引
        last_lat_idx = (self.sample_lat_size_dis - 1) // stride_h * stride_h
        last_lon_idx = (self.sample_lon_size_dis - 1) // stride_w * stride_w

        self.max_indices = (last_lat_idx, last_lon_idx)
        print('self.max_indices', self.max_indices)

    def init_lat_weight(self):
        if not self.args.compute_mean_std:
            self.mean_std_era5 = torch.from_numpy(
                np.load(os.path.join(self.results_dir, 'mean_std_era5_391.npy'))).type(
                torch.float32).to(self.device)

    def test(self, mode='val', test_epoch=500):

        checkpoint_path = os.path.join(
            self.args.save_dir,
            f'checkpoint_epoch{test_epoch}.pth'
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"unexist: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 1. 恢复模型参数
        self.model.load_state_dict(checkpoint['model'])

        six_hour_accum = False
        if six_hour_accum:
            optimal_prob_thresholds = self.optimize_on_validation_6hour_accum(
                self.dataloader_test, self.device, self.args, mode='val',
                                   search_resolution=self.args.search_resolution,
            )
            i = 0
            while os.path.exists(os.path.join(self.results_dir, 'optimal_thre', self.args.ex_name,
                                              'optimal_prob_thresholds_6hour_accum_epoch_' + str(test_epoch) + '_' + str(i) + '.npy')):
                i += 1
            np.save(os.path.join(self.results_dir, 'optimal_thre',
                                 self.args.ex_name, 'optimal_prob_thresholds_6hour_accum_epoch_'+str(test_epoch) + '_' + str(i) + '.npy'), optimal_prob_thresholds)
            # self.optimal_prob_thresholds = np.load(os.path.join(self.results_dir, 'optimal_thre',
            #                      self.args.ex_name, 'optimal_prob_thresholds_6hour_accum_epoch_'+str(test_epoch)+ '_' + str(0) + '.npy'))
            self.evaluate_6hour_accum(
                metric_list=['mae', 'rmse'], mode=mode, test_epoch=test_epoch  # , thre=i  # the validation dataset
            )
        else:
            # optimal_prob_thresholds = self.optimize_on_validation(
            #     self.dataloader_val, self.device, self.args, mode='val',
            #     search_resolution=self.args.search_resolution,
            # )
            # i = 0
            # while os.path.exists(os.path.join(self.results_dir, 'optimal_thre', self.args.ex_name,
            #                                   'optimal_prob_thresholds_epoch' + str(test_epoch) + '_' + str(i) + '.npy')):
            #     i += 1
            # np.save(os.path.join(self.results_dir, 'optimal_thre',
            #                      self.args.ex_name,
            #                      'optimal_prob_thresholds_epoch' + str(test_epoch) + '_' + str(i) + '.npy'),
            #         optimal_prob_thresholds)

            self.optimal_prob_thresholds = np.load(os.path.join(self.results_dir, 'optimal_thre',
                                 self.args.ex_name, 'optimal_prob_thresholds_epoch'+str(test_epoch)+ '_' + str(0) + '.npy'))
            self.evaluate(
                metric_list=['mae', 'rmse'], mode=mode, test_epoch=test_epoch#, thre=i  # the validation dataset
            )

    def merge_patches_efficient(self, patches, original_shape):
        """
        高效版本的patch拼接
        """
        bs, bs_p, time_step, ch, patch_h, patch_w = patches.shape

        # 重塑patches以便批量处理
        patches_reshaped = patches.permute(0, 2, 3, 4, 5, 1)  # bs, bs_p, time_step, ch, patch_h, patch_w

        # 使用fold操作进行拼接（更高效）
        # self.args.tar_size, self.args.crop_stride_test
        full_images = F.fold(
            patches_reshaped.reshape(bs * time_step, -1, bs_p),
            output_size=original_shape,
            kernel_size=self.args.tar_L1_shape[-2:],
            stride=self.args.crop_stride_test,
            padding=0
        ).reshape(bs, time_step, ch, original_shape[0], original_shape[1])
        # print(full_images.shape)
        # print(full_images[0, 0, 0])

        # # 计算重叠权重
        ones_patches = torch.ones_like(patches_reshaped.reshape(bs * time_step, -1, bs_p))
        weight_maps = F.fold(
            ones_patches,
            output_size=original_shape,
            kernel_size=self.args.tar_L1_shape[-2:],
            stride=self.args.crop_stride_test,
            padding=0
        ).reshape(bs, time_step, ch, original_shape[0], original_shape[1])
        # print(weight_maps[0, 0, 0])
        result = full_images / weight_maps
        # print(result[0, 0, 0])

        return result

    def evaluate(self, epoch=None, metric_list=['mae', 'mse', 'rmse', 'ssim'], mode='val', test_epoch=100):

        forcast_len = self.args.aft_seq_length_test
        dataloader = self.dataloader_test
        # self.args.aft_seq_length = self.args.aft_seq_length_test
        spatial_norm = True
        self.model.eval()
        eval_res_list = []
        mean = []
        std = []
        time_list = []
        all_means = []
        all_stds = []
        all_labels = []  # 原始降水率标签
        all_masks = []
        margin_wid = 2
        from datetime import datetime
        # 获取当前时间
        now = datetime.now()
        print(len(dataloader))
        CRPS_res_list = []
        fss_neighborhood = getattr(self.args, 'fss_neighborhood', 5)
        with torch.no_grad():
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data, rqi_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)
                    # print(L1_data.shape)
                # logging.info(str(step))  # 自动记录时间和内容
                # logging.info(time_str)  # 自动记录时间和内容
                # print(qpe_data_full.shape)
                # print(rqi_data.shape)
                # print(qpe_data.shape)
                # print(time_data.shape)
                # print(coords_data.shape)
                # bs, _, _, _, _ = L1_data.shape
                # print(L1_data.shape)
                bs, bs_p, _, _, _, _ = L1_data.shape
                # print(L1_data.shape)
                L1_data = L1_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0, 1).type(torch.float32).to(self.device,
                                                                           non_blocking=True)
                rqi_data = rqi_data[:, :, None, ...].type(torch.float32).to(self.device, non_blocking=True)
                # print(rqi_data.shape)
                # print('11111111')
                # print(era5_data.shape)
                # print(era5_data.device)
                era5_data = torch.nan_to_num(era5_data, nan=0.0)
                # era5_data = (era5_data - self.mean_std_era5[0][None, None, :, None, None]) / self.mean_std_era5[1][None,
                #                                                                              None, :, None, None]
                # # print(era5_data.shape)
                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_mask_full = 1 - (torch.isnan(qpe_data_full) | (qpe_data_full < 0) | (qpe_data_full > self.args.max_qpe)).float()
                valid_mask = qpe_mask_full[:, self.args.in_len_val:].type(torch.int)
                valid_mask = valid_mask * ((rqi_data > self.args.rqi_thre).int())
                # print((rqi_data > thre).int().sum()/ torch.ones_like(rqi_data).sum())
                # valid_mask要乘以对应时刻的RQI>0.5/0.6/0.7/0.8/0.9/
                qpe_data_full = torch.nan_to_num(qpe_data_full, nan=0.0).clamp(min=0.0)
                qpe_data_full[qpe_data_full > self.args.max_qpe] = 0.0
                labels = qpe_data_full[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()
                # true_class = torch.where(labels > self.args.ori_thre, torch.tensor(1.0), torch.tensor(0.0)).type(
                #     torch.int)

                qpe_data = torch.nan_to_num(qpe_data, nan=0.0).clamp_(min=0.0)
                qpe_data[qpe_data > self.args.max_qpe] = 0.0
                # labels = qpe_data[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()
                qpe_data = (qpe_data - self.args.max_min_qpe[1]) / (self.args.max_min_qpe[0] - self.args.max_min_qpe[1])

                L1_data = torch.nan_to_num(L1_data, nan=0.0).clamp_(min=0.0)
                # L1_data = (L1_data - self.args.max_min[1]) / (self.args.max_min[0] - self.args.max_min[1])
                L1_data.sub_(self.args.max_min[1])
                L1_data.div_(self.args.max_min[0] - self.args.max_min[1])

                inputs = torch.cat([L1_data, qpe_data[:, :self.args.in_len_val,
                                             ...]], dim=2)  # .type(torch.float32).to(self.device, non_blocking=True)

                time_data = time_data.type(torch.float32).repeat(inputs.shape[0], 1, 1).to(self.device,
                                                                                           non_blocking=True)

                geo_coords = torch.stack([self.geo_coords[:,
                                          lat_idx:lat_idx + self.coords_seg_size[0],
                                          lon_idx:lon_idx + self.coords_seg_size[1]] for
                                          (lat_idx, lon_idx) in self.test_indices]).to(non_blocking=True)
                dem_data = torch.stack([self.dem_data[:, int(
                    self.args.dem_ratio[0] * (lat_idx)):int(
                    self.args.dem_ratio[0] * (lat_idx + self.coords_seg_size[0])),
                                        int(self.args.dem_ratio[1] * (lon_idx)):int(
                                            self.args.dem_ratio[1] * (
                                                        lon_idx + self.coords_seg_size[1]))]
                                        for (lat_idx, lon_idx) in self.test_indices]).to(
                    non_blocking=True)

                time0 = time.time()

                if 0:
                    inputs = inputs.half()
                    era5_data = era5_data.half()
                    time_data = time_data.half()
                # print(inputs.shape)
                # print(era5_data.shape)
                # print(time_data.shape)
                # # print(time_data.shape)
                # print(coords_data.shape)
                chunk_size = 1
                pred, pred_class = torch.empty(0, forcast_len, *qpe_data.shape[-3:]) \
                    , torch.empty(0, forcast_len, *qpe_data.shape[-3:])  # .to(self.device)
                for i in range(int(inputs.shape[0] / chunk_size)):
                    # print(i)
                    pred_tmp, pred_std_tmp = self.model(inputs[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          era5_data[i * chunk_size:(i + 1) * chunk_size].clone()
                                                          , self.const_data,
                                                          time_data[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          geo_coords=geo_coords[
                                                                     i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          dem_data=dem_data[
                                                                   i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          # labels_qpe=labels,
                                                          aft_seq_length=forcast_len,
                                                          shrink=self.args.shrink, mode=mode,
                                                          device=self.device)
                    pred = torch.cat([pred, pred_tmp.cpu()])
                    pred_class = torch.cat([pred_class, pred_std_tmp.cpu()])
                    # del pred_tmp, pred_class_tmp
                    # import gc
                    # gc.collect()
                    # torch.cuda.empty_cache()
                pred_mean = self.merge_patches_efficient(
                    pred.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )
                pred_std = self.merge_patches_efficient(
                    pred_class.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )

                time_list.append((time.time() - time0))
                # print(pred_mean.shape)

                all_means.append(pred_mean[:, :, 0].clone().cpu())
                all_stds.append(pred_std[:, :, 0].clone().cpu())
                os.makedirs(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name), exist_ok=True)
                np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                                     f'pred_{step}.npy'), torch.cat([pred_mean, pred_std], 2).cpu().numpy())
                np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                                     f'label_{step}.npy'),
                        labels[:, :, 0, self.args.border_tar[0] + self.args.lat_dismiss[0]
                                        :pred_mean.shape[-2] + self.args.border_tar[0] + self.args.lat_dismiss[0],
                        self.args.border_tar[1] + self.args.lon_dismiss[0]
                        :pred_mean.shape[-1] + self.args.border_tar[1] + self.args.lon_dismiss[
                            0]].clone().cpu().numpy())

                all_labels.append(labels[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())
                all_masks.append(valid_mask[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())

                # print(pred_mean.shape)
                # print(pred_std.shape)
                # print(labels.shape)
                # print(valid_mask.shape)

                # time_list.append((time.time() - time0))

                # del labels, last_res, valid_mask, true_class
                # import gc
                # gc.collect()
                # torch.cuda.empty_cache()
                # 暂时注销
                if self.args.empty_cache:
                    torch.cuda.empty_cache()

                CRPS_res = compute_crps_fully_vectorized(
                    pred_mean.cpu(), pred_std.cpu(),
                    labels[:, :, :, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].cpu(),
                    valid_mask[:, :, :, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].cpu()
                )
                CRPS_res_list.append(CRPS_res)
                # print(CRPS_res)
                # exit()
                # 清空 GPU 缓存
                del labels, pred_mean, pred_std, valid_mask
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # 强制垃圾回收
            # exit()
            # 合并所有batch
            all_means = torch.cat(all_means, dim=0)
            all_stds = torch.cat(all_stds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            # 展平空间维度
            N, T, H, W = all_means.shape
            print(all_means.shape)
            all_means_flat = all_means.permute(0, 2, 3, 1).flatten(0, 2)  # [N*H*W, T]
            all_stds_flat = all_stds.permute(0, 2, 3, 1).flatten(0, 2)
            all_labels_flat = all_labels.permute(0, 2, 3, 1).flatten(0, 2)
            all_masks_flat = all_masks.permute(0, 2, 3, 1).flatten(0, 2)  # [N*H*W, T]
            print(all_means_flat.shape)

            # 计算CSI指标
            # 计算CSI, POD, FAR指标
            csi_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds)])
            pod_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds)])
            far_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds)])
            fss_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds)])

            for i, pt in enumerate(self.precip_thresholds):
                # 计算 P(rain >= pt)
                probs = self._compute_prob_exceed_threshold(all_means_flat, all_stds_flat, pt)
                labels_binary = (all_labels_flat >= pt).float()

                # 新增：基于最优概率阈值，将概率预报确定化为二值事件场后计算 FSS。
                # all_means_flat/all_stds_flat: [N*H*W, T]
                # probs: [N*H*W, T] -> prob_maps: [N, T, H, W]
                prob_maps = probs.reshape(N, H, W, T).permute(0, 3, 1, 2)
                label_binary_maps = labels_binary.reshape(N, H, W, T).permute(0, 3, 1, 2)

                # 每个预报时效使用对应的最优概率阈值：
                # P(rain >= pt) > optimal_prob_thresholds[t, i] 时，判定降水事件发生。
                optimal_thresholds = torch.as_tensor(
                    self.optimal_prob_thresholds[:, i],
                    dtype=prob_maps.dtype,
                    device=prob_maps.device
                ).view(1, T, 1, 1)
                pred_binary_maps = (prob_maps > optimal_thresholds).float()

                # calc_fss 的输入格式为 [T, B, C, H, W]。
                # 这里 pred/target 已是二值事件场，因此统一使用 0.5 作为二值场阈值。
                fss_all_i = self.calc_fss(
                    pred=pred_binary_maps.permute(1, 0, 2, 3).unsqueeze(2).cpu().numpy(),
                    target=label_binary_maps.permute(1, 0, 2, 3).unsqueeze(2).cpu().numpy(),
                    mask=all_masks.permute(1, 0, 2, 3).unsqueeze(2).cpu().numpy(),
                    thresholds=self.args.threholds,
                    neighborhood=fss_neighborhood
                )

                # fss_all_i shape = [T, N, C, 1] -> [T]
                fss_results[:, i] = np.nanmean(fss_all_i[:, :, :, 0], axis=(1, 2))

                for t in range(T):
                    prob_thresh = self.optimal_prob_thresholds[t, i]
                    current_probs = probs[:, t]
                    current_labels = labels_binary[:, t]
                    current_mask = all_masks_flat[:, t]

                    csi = self._compute_csi_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )
                    pod = self._compute_pod_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )
                    far = self._compute_far_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )

                    csi_results[t, i] = csi
                    pod_results[t, i] = pod
                    far_results[t, i] = far

                    if 1:
                        valid_count = (current_mask == 1).sum().item()
                        print(f"Test: lead_time={t + 1}, precip_thresh={pt}mm/h, "
                              f"prob_thresh={prob_thresh:.3f}, "
                              f"CSI={csi:.4f}, POD={pod:.4f}, FAR={far:.4f}, "
                              f"FSS={fss_results[t, i]:.4f}, "
                              f"valid_samples={valid_count}")

            # print(csi_results)

            # self.print_info()
            # valid_pod, valid_far, valid_csi, valid_hss, valid_acc, valid_mse, valid_mae, valid_balanced_mse, \
            # valid_balanced_mae, freq_bias = self.evaluate_metric.calculate_stat()
            # print(CRPS_res_list)
            valid_crps = torch.stack(CRPS_res_list).nanmean(0).numpy()
            valid_fss = fss_results

            print('FSS precip thresholds:', self.precip_thresholds)
            print('FSS optimal probability thresholds:', self.optimal_prob_thresholds)
            print('FSS shape:', valid_fss.shape)
            print('FSS:', valid_fss)
            # print(valid_crps)
            # print(valid_pod.shape)
            # print(valid_mse.shape)
            # print(valid_balanced_mse.shape)
            # exit()

            # final_crps = self.CRPS_METRIC.compute()
            # mean_crps = self.CRPS_METRIC.compute_mean()
            print('mean time:', np.mean(time_list))
            if epoch is not None:
                pkl_path = os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                        'metric_epoch' + str(epoch) + '.pkl')
            else:
                i = 0
                while os.path.exists(os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                                  'metric_' + str(test_epoch) + '_' + str(i) + '.pkl')):
                    i += 1
                pkl_path = os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                        'metric_' + str(test_epoch) + '_' + str(i) + '.pkl')
            with open(pkl_path, 'wb') as file:
                pickle.dump(
                    [pod_results, far_results, csi_results, valid_crps, valid_fss], file)
            # self.plot_result(
            #     [pod_results, far_results, csi_results, valid_crps],
            #     epoch=epoch, test_epoch=test_epoch)
            # exit()

        # self.evaluate_metric.clear_all()  # 所有指标归零

        # np.set_printoptions(precision=2, suppress=True)
        # return np.mean(pod_results, 1), np.mean(far_results, 1), np.mean(csi_results, 1), np.mean(valid_crps, 1)

    def calc_fss(self, pred, target, mask=None, thresholds=None, neighborhood=9):
        """
        计算带 mask 的 Fractions Skill Score。

        参数
        ----------
        pred : np.ndarray
            预测降水，shape = [T, B, C, H, W]，单位必须是原始降水量。
        target : np.ndarray
            真实降水，shape = [T, B, C, H, W]，单位必须是原始降水量。
        mask : np.ndarray or None
            有效区域，shape = [T, B, 1, H, W] 或 [T, B, C, H, W]。
            1 表示有效，0 表示无效。
        thresholds : array-like
            FSS 阈值，例如 [0.2, 1, 2, 4, 8, 20]。
        neighborhood : int
            空间邻域大小，例如 9 表示 9×9 邻域。

        返回
        ----------
        fss : np.ndarray
            shape = [T, B, C, threshold_num]
        """
        if thresholds is None:
            raise ValueError("thresholds 不能为空。")

        if neighborhood <= 0 or neighborhood % 2 == 0:
            raise ValueError("FSS neighborhood 必须是正奇数，例如 3、5、7、9。")

        thresholds = np.asarray(thresholds, dtype=np.float32)

        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        if mask is None:
            mask = np.ones_like(target, dtype=np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
            mask = np.broadcast_to(mask, target.shape)

        mask = (mask > 0).astype(np.float32)

        threshold_num = len(thresholds)
        fss = np.full(target.shape[:-2] + (threshold_num,), np.nan,
                      dtype=np.float32)

        window_area = float(neighborhood * neighborhood)

        # 每个中心点周围的有效像素数量。
        # 边界外的位置视为无效，因此边界邻域会按照实际有效像素数归一化。
        valid_count = uniform_filter(
            mask,
            size=neighborhood,
            axes=(-2, -1),
            mode='constant',
            cval=0.0
        ) * window_area

        # FSS 最终只在中心像素有效的位置聚合。
        center_valid_count = np.sum(mask, axis=(-2, -1))

        for threshold_idx, threshold in enumerate(thresholds):
            # 与现有 CSI 的事件定义保持一致，这里使用大于阈值。
            obs_event = (target > threshold).astype(np.float32)
            pred_event = (pred > threshold).astype(np.float32)

            # 无效像素既不能作为有雨，也不能作为无雨参与邻域统计。
            obs_count = uniform_filter(
                obs_event * mask,
                size=neighborhood,
                axes=(-2, -1),
                mode='constant',
                cval=0.0
            ) * window_area

            pred_count = uniform_filter(
                pred_event * mask,
                size=neighborhood,
                axes=(-2, -1),
                mode='constant',
                cval=0.0
            ) * window_area

            # 观测和预测的邻域降水事件发生比例。
            obs_fraction = np.zeros_like(obs_count, dtype=np.float32)
            pred_fraction = np.zeros_like(pred_count, dtype=np.float32)

            np.divide(
                obs_count,
                valid_count,
                out=obs_fraction,
                where=valid_count > 0
            )
            np.divide(
                pred_count,
                valid_count,
                out=pred_fraction,
                where=valid_count > 0
            )

            # FSS:
            # 1 - mean((Fo - Fp)^2) / mean(Fo^2 + Fp^2)
            fraction_mse = np.sum(
                np.square(obs_fraction - pred_fraction) * mask,
                axis=(-2, -1)
            )
            reference = np.sum(
                (np.square(obs_fraction) + np.square(pred_fraction)) * mask,
                axis=(-2, -1)
            )

            score = np.full(reference.shape, np.nan, dtype=np.float32)

            has_valid_center = center_valid_count > 0
            has_event = reference > 0

            # 观测和预测在该阈值下都没有事件时，认为完全匹配。
            score[has_valid_center & (~has_event)] = 1.0

            normal_index = has_valid_center & has_event
            score[normal_index] = (
                1.0
                - fraction_mse[normal_index] / reference[normal_index]
            )

            fss[..., threshold_idx] = np.clip(score, 0.0, 1.0)

        return fss

    def _compute_csi_from_probs(self, probs, labels, prob_threshold, mask=None):
        """
        基于概率阈值计算CSI，支持mask
        使用PyTorch在CPU上计算

        Args:
            probs: 预测概率 (降水率 >= r 的概率), shape [N, lead_times] 或 [N]
            labels: 真实标签 (降水率 >= r), shape [N, lead_times] 或 [N]
            prob_threshold: 概率阈值 (0-1)
            mask: 有效样本mask, shape [N, lead_times] 或 [N], 1表示有效样本, 0表示无效

        Returns:
            CSI值 (float)
        """
        # 确保是PyTorch Tensor并在CPU上
        if not torch.is_tensor(probs):
            probs = torch.from_numpy(probs) if isinstance(probs, np.ndarray) else probs
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

        probs = probs.cpu()
        labels = labels.cpu()

        # 应用概率阈值得到二元预测
        pred_binary = (probs >= prob_threshold).float()

        # 应用mask
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
            mask = mask.cpu()

            # 只保留mask=1的样本
            valid_indices = mask == 1

            # 如果没有有效样本，返回0
            if valid_indices.sum() == 0:
                return 0.0

            pred_binary = pred_binary[valid_indices]
            labels = labels[valid_indices]

        # 计算TP, FP, FN
        TP = ((pred_binary == 1) & (labels == 1)).sum().item()
        FP = ((pred_binary == 1) & (labels == 0)).sum().item()
        FN = ((pred_binary == 0) & (labels == 1)).sum().item()

        if TP + FP + FN == 0:
            return 0.0

        return TP / (TP + FP + FN)

    def _compute_pod_from_probs(self, probs, labels, prob_threshold, mask=None):
        """
        基于概率阈值计算POD (Probability of Detection)，支持mask
        POD = TP / (TP + FN)
        使用PyTorch在CPU上计算

        Args:
            probs: 预测概率, shape [N] 或 [N, lead_times]
            labels: 真实标签 (0/1), shape [N] 或 [N, lead_times]
            prob_threshold: 概率阈值 (0-1)
            mask: 有效样本mask, shape与probs相同, 1表示有效, 0表示无效

        Returns:
            POD值 (float)
        """
        # 确保是PyTorch Tensor并在CPU上
        if not torch.is_tensor(probs):
            probs = torch.from_numpy(probs) if isinstance(probs, np.ndarray) else probs
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

        probs = probs.cpu()
        labels = labels.cpu()

        # 应用概率阈值得到二元预测
        pred_binary = (probs >= prob_threshold).float()

        # 应用mask
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
            mask = mask.cpu()
            valid_indices = mask == 1
            if valid_indices.sum() == 0:
                return 0.0
            pred_binary = pred_binary[valid_indices]
            labels = labels[valid_indices]

        # 计算TP, FN
        TP = ((pred_binary == 1) & (labels == 1)).sum().item()
        FN = ((pred_binary == 0) & (labels == 1)).sum().item()

        if TP + FN == 0:
            return 0.0

        return TP / (TP + FN)

    def _compute_far_from_probs(self, probs, labels, prob_threshold, mask=None):
        """
        基于概率阈值计算FAR (False Alarm Ratio)，支持mask
        FAR = FP / (TP + FP)
        使用PyTorch在CPU上计算

        Args:
            probs: 预测概率, shape [N] 或 [N, lead_times]
            labels: 真实标签 (0/1), shape [N] 或 [N, lead_times]
            prob_threshold: 概率阈值 (0-1)
            mask: 有效样本mask, shape与probs相同, 1表示有效, 0表示无效

        Returns:
            FAR值 (float)
        """
        # 确保是PyTorch Tensor并在CPU上
        if not torch.is_tensor(probs):
            probs = torch.from_numpy(probs) if isinstance(probs, np.ndarray) else probs
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

        probs = probs.cpu()
        labels = labels.cpu()

        # 应用概率阈值得到二元预测
        pred_binary = (probs >= prob_threshold).float()

        # 应用mask
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
            mask = mask.cpu()
            valid_indices = mask == 1
            if valid_indices.sum() == 0:
                return 0.0
            pred_binary = pred_binary[valid_indices]
            labels = labels[valid_indices]

        # 计算TP, FP
        TP = ((pred_binary == 1) & (labels == 1)).sum().item()
        FP = ((pred_binary == 1) & (labels == 0)).sum().item()

        if TP + FP == 0:
            return 0.0

        return FP / (TP + FP)

    def _compute_prob_exceed_threshold(self, mean, std, precip_threshold):
        """
        基于高斯分布计算降水率超过给定阈值的概率
        P(rain >= r) = 1 - Φ((r - μ)/σ)
        使用PyTorch在CPU上计算

        Args:
            mean: 预测均值 [N, lead_times, H, W] (PyTorch Tensor)
            std: 预测标准差 [N, lead_times, H, W] (PyTorch Tensor)
            precip_threshold: 降水强度阈值 (float)

        Returns:
            prob: 超过阈值的概率 [N, lead_times, H, W] (PyTorch Tensor)
        """
        # 确保在CPU上
        mean = mean.cpu()
        std = std.cpu()

        # 使用PyTorch的标准正态分布CDF
        # 标准化
        z_score = (precip_threshold - mean) / (std + 1e-8)
        # P(rain >= r) = 1 - CDF(z)
        # 使用torch.distributions.normal.Normal的cdf方法
        from torch.distributions import Normal
        normal_dist = Normal(0, 1)
        prob = 1 - normal_dist.cdf(z_score)

        # 处理极端值
        prob = torch.clamp(prob, min=0.0, max=1.0)

        return prob

    def optimize_on_validation(self, val_dataloader, device, args,
                               search_resolution=50, mode='val', verbose=True):
        """
        在验证集上优化概率阈值
        使用PyTorch在CPU上计算

        Args:
            val_dataloader: 验证集数据加载器
            device: 设备（用于模型推理）
            args: 参数配置
            search_resolution: 概率阈值搜索粒度 (0到1之间搜索search_resolution个点)
            verbose: 是否打印进度

        Returns:
            optimal_thresholds: 最优概率阈值 [lead_times, len(precip_thresholds)]
        """
        self.model.eval()
        forcast_len = args.aft_seq_length_test

        # 收集所有预测和标签
        print("Collecting predictions and labels on validation set...")
        all_means = []
        all_stds = []
        all_labels = []  # 原始降水率标签
        all_masks = []
        time_list = []

        with torch.no_grad():
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data, rqi_data) in enumerate(
                    tqdm(val_dataloader, disable=not verbose)):

                bs, bs_p, _, _, _, _ = L1_data.shape

                L1_data = L1_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0, 1).type(torch.float32).to(self.device,
                                                                           non_blocking=True)
                rqi_data = rqi_data[:, :, None, ...].type(torch.float32).to(self.device, non_blocking=True)

                era5_data = torch.nan_to_num(era5_data, nan=0.0)
                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_mask_full = 1 - (torch.isnan(qpe_data_full) | (qpe_data_full < 0) | (qpe_data_full > self.args.max_qpe)).float()
                valid_mask = qpe_mask_full[:, self.args.in_len_val:].type(torch.int)
                valid_mask = valid_mask * ((rqi_data > self.args.rqi_thre).int())
                print(torch.mean(valid_mask.float()))
                qpe_data_full = torch.nan_to_num(qpe_data_full, nan=0.0).clamp(min=0.0)
                qpe_data_full[qpe_data_full > self.args.max_qpe] = 0.0
                labels = qpe_data_full[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()

                qpe_data = torch.nan_to_num(qpe_data, nan=0.0).clamp_(min=0.0)
                qpe_data[qpe_data > self.args.max_qpe] = 0.0
                qpe_data = (qpe_data - self.args.max_min_qpe[1]) / (self.args.max_min_qpe[0] - self.args.max_min_qpe[1])

                L1_data = torch.nan_to_num(L1_data, nan=0.0).clamp_(min=0.0)
                L1_data.sub_(self.args.max_min[1])
                L1_data.div_(self.args.max_min[0] - self.args.max_min[1])

                inputs = torch.cat([L1_data, qpe_data[:, :self.args.in_len_val,
                                             ...]], dim=2)  # .type(torch.float32).to(self.device, non_blocking=True)

                time_data = time_data.type(torch.float32).repeat(inputs.shape[0], 1, 1).to(self.device,
                                                                                           non_blocking=True)

                geo_coords = torch.stack([self.geo_coords[:,
                                          lat_idx:lat_idx + self.coords_seg_size[0],
                                          lon_idx:lon_idx + self.coords_seg_size[1]] for
                                          (lat_idx, lon_idx) in self.test_indices]).to(non_blocking=True)
                dem_data = torch.stack([self.dem_data[:, int(
                    self.args.dem_ratio[0] * (lat_idx)):int(
                    self.args.dem_ratio[0] * (lat_idx + self.coords_seg_size[0])),
                                        int(self.args.dem_ratio[1] * (lon_idx)):int(
                                            self.args.dem_ratio[1] * (
                                                    lon_idx + self.coords_seg_size[1]))]
                                        for (lat_idx, lon_idx) in self.test_indices]).to(
                    non_blocking=True)

                time0 = time.time()

                if 0:
                    inputs = inputs.half()
                    era5_data = era5_data.half()
                    time_data = time_data.half()

                chunk_size = 1
                pred, pred_class = torch.empty(0, forcast_len, *qpe_data.shape[-3:]) \
                    , torch.empty(0, forcast_len, *qpe_data.shape[-3:])  # .to(self.device)
                for i in range(int(inputs.shape[0] / chunk_size)):
                    # print(i)
                    pred_tmp, pred_std_tmp = self.model(inputs[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        era5_data[i * chunk_size:(i + 1) * chunk_size].clone()
                                                        , self.const_data,
                                                        time_data[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        geo_coords=geo_coords[
                                                                   i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        dem_data=dem_data[
                                                                 i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        # labels_qpe=labels,
                                                        aft_seq_length=forcast_len,
                                                        shrink=self.args.shrink, mode=mode,
                                                        device=self.device)
                    pred = torch.cat([pred, pred_tmp.cpu()])
                    pred_class = torch.cat([pred_class, pred_std_tmp.cpu()])

                pred_mean = self.merge_patches_efficient(
                    pred.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )
                pred_std = self.merge_patches_efficient(
                    pred_class.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )

                time_list.append((time.time() - time0))

                all_means.append(pred_mean[:, :, 0].clone().cpu())
                all_stds.append(pred_std[:, :, 0].clone().cpu())
                all_labels.append(labels[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())
                all_masks.append(valid_mask[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())

        # 合并所有batch（已在CPU上）
        all_means = torch.cat(all_means, dim=0)  # [N, T, H, W]
        all_stds = torch.cat(all_stds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)  # [N, T, H, W]

        # 展平空间维度
        N, T, H, W = all_means.shape
        print(all_means.shape)
        all_means_flat = all_means.permute(0,2,3,1).flatten(0, 2)  # [N*H*W, T]
        all_stds_flat = all_stds.permute(0,2,3,1).flatten(0, 2)
        all_labels_flat = all_labels.permute(0,2,3,1).flatten(0, 2)
        all_masks_flat = all_masks.permute(0,2,3,1).flatten(0, 2)   # [N*H*W, T]
        print(all_means_flat.shape)

        # 为每个降水强度阈值计算概率和标签
        print("Computing probabilities for each precipitation threshold...")
        all_probs_dict = {}
        all_labels_dict = {}
        all_masks_dict = {}  # 添加mask字典

        for pt in self.precip_thresholds:
            # 计算 P(rain >= pt) (返回PyTorch Tensor)
            probs = self._compute_prob_exceed_threshold(all_means_flat, all_stds_flat, pt)
            all_probs_dict[pt] = probs  # [N*H*W, T]

            # 生成真实标签 (PyTorch Tensor)
            labels_binary = (all_labels_flat >= pt).float()
            all_labels_dict[pt] = labels_binary

            # mask保持不变（所有强度阈值共用同一个mask）
            all_masks_dict[pt] = all_masks_flat  # [N*H*W, T]

        # 搜索最优概率阈值
        print(f"Searching optimal probability thresholds (resolution={search_resolution})...")

        # 初始化最优阈值存储
        optimal_thresholds = np.zeros((self.lead_times, len(self.precip_thresholds)))

        # 对每个降水强度阈值和每个预测时刻分别优化
        for i, pt in enumerate(self.precip_thresholds):
            probs = all_probs_dict[pt]  # [N_samples, T]
            labels = all_labels_dict[pt]
            masks = all_masks_dict[pt]  # [N_samples, T]

            for t in range(self.lead_times):
                # 候选概率阈值
                if i < 5:
                    candidate_thresholds = np.linspace(0.01, 0.99, search_resolution)
                elif i == 5:
                    candidate_thresholds = np.linspace(0.002, 0.2, search_resolution)
                best_csi = 0.0
                best_thresh = 0.5  # 默认阈值

                # 获取当前时刻的mask
                current_mask = masks[:, t]  # [N_samples]

                # 检查是否有有效样本
                if (current_mask == 1).sum().item() == 0:
                    if verbose:
                        print(f"Warning: No valid samples for pt={pt}, t={t}, using default threshold=0.5")
                    optimal_thresholds[t, i] = 0.5
                    continue

                # 获取当前时刻的概率和标签
                current_probs = probs[:, t]  # [N_samples]
                current_labels = labels[:, t]  # [N_samples]

                # 搜索最优阈值
                for thresh in candidate_thresholds:
                    csi = self._compute_csi_from_probs(
                        current_probs,
                        current_labels,
                        thresh,
                        mask=current_mask
                    )
                    if csi > best_csi:
                        best_csi = csi
                        best_thresh = thresh

                optimal_thresholds[t, i] = best_thresh

                if verbose:
                    valid_count = (current_mask == 1).sum().item()
                    print(f"Precip threshold={pt}mm/h, lead_time={t + 1}: "
                          f"best_thresh={best_thresh:.3f}, best_CSI={best_csi:.4f}, "
                          f"valid_samples={valid_count}")

        self.optimal_prob_thresholds = optimal_thresholds

        return optimal_thresholds

    def evaluate_6hour_accum(self, epoch=None, metric_list=['mae', 'mse', 'rmse', 'ssim'], mode='val', test_epoch=100):

        forcast_len = self.args.aft_seq_length_test
        dataloader = self.dataloader_test
        spatial_norm = True
        self.model.eval()
        eval_res_list = []
        mean = []
        std = []
        time_list = []
        all_means = []
        all_stds = []
        all_labels = []
        all_masks = []
        margin_wid = 2
        from datetime import datetime
        now = datetime.now()
        print(len(dataloader))
        CRPS_res_list = []
        with torch.no_grad():
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data, rqi_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)

                bs, bs_p, _, _, _, _ = L1_data.shape

                L1_data = L1_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0, 1).type(torch.float32).to(self.device,
                                                                           non_blocking=True)
                rqi_data = rqi_data[:, :, None, ...].type(torch.float32).to(self.device, non_blocking=True)

                era5_data = torch.nan_to_num(era5_data, nan=0.0)
                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_mask_full = 1 - (torch.isnan(qpe_data_full) | (qpe_data_full < 0) | (qpe_data_full > self.args.max_qpe)).float()
                valid_mask = qpe_mask_full[:, self.args.in_len_val:].type(torch.int)
                valid_mask = valid_mask * ((rqi_data > self.args.rqi_thre).int())
                qpe_data_full = torch.nan_to_num(qpe_data_full, nan=0.0).clamp(min=0.0)
                qpe_data_full[qpe_data_full > self.args.max_qpe] = 0.0
                labels = qpe_data_full[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()

                qpe_data = torch.nan_to_num(qpe_data, nan=0.0).clamp_(min=0.0)
                qpe_data[qpe_data > self.args.max_qpe] = 0.0
                qpe_data = (qpe_data - self.args.max_min_qpe[1]) / (self.args.max_min_qpe[0] - self.args.max_min_qpe[1])

                L1_data = torch.nan_to_num(L1_data, nan=0.0).clamp_(min=0.0)
                L1_data.sub_(self.args.max_min[1])
                L1_data.div_(self.args.max_min[0] - self.args.max_min[1])

                inputs = torch.cat([L1_data, qpe_data[:, :self.args.in_len_val,
                                             ...]], dim=2)  # .type(torch.float32).to(self.device, non_blocking=True)

                time_data = time_data.type(torch.float32).repeat(inputs.shape[0], 1, 1).to(self.device,
                                                                                           non_blocking=True)

                geo_coords = torch.stack([self.geo_coords[:,
                                          lat_idx:lat_idx + self.coords_seg_size[0],
                                          lon_idx:lon_idx + self.coords_seg_size[1]] for
                                          (lat_idx, lon_idx) in self.test_indices]).to(non_blocking=True)
                dem_data = torch.stack([self.dem_data[:, int(
                    self.args.dem_ratio[0] * (lat_idx)):int(
                    self.args.dem_ratio[0] * (lat_idx + self.coords_seg_size[0])),
                                        int(self.args.dem_ratio[1] * (lon_idx)):int(
                                            self.args.dem_ratio[1] * (
                                                        lon_idx + self.coords_seg_size[1]))]
                                        for (lat_idx, lon_idx) in self.test_indices]).to(
                    non_blocking=True)

                time0 = time.time()

                if 0:
                    inputs = inputs.half()
                    era5_data = era5_data.half()
                    time_data = time_data.half()

                chunk_size = 1
                pred, pred_class = torch.empty(0, forcast_len, *qpe_data.shape[-3:]) \
                    , torch.empty(0, forcast_len, *qpe_data.shape[-3:])  # .to(self.device)
                for i in range(int(inputs.shape[0] / chunk_size)):
                    # print(i)
                    pred_tmp, pred_std_tmp = self.model(inputs[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          era5_data[i * chunk_size:(i + 1) * chunk_size].clone()
                                                          , self.const_data,
                                                          time_data[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          geo_coords=geo_coords[
                                                                     i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          dem_data=dem_data[
                                                                   i * chunk_size:(i + 1) * chunk_size].clone(),
                                                          # labels_qpe=labels,
                                                          aft_seq_length=forcast_len,
                                                          shrink=self.args.shrink, mode=mode,
                                                          device=self.device)
                    pred = torch.cat([pred, pred_tmp.cpu()])
                    pred_class = torch.cat([pred_class, pred_std_tmp.cpu()])

                pred_mean = self.merge_patches_efficient(
                    pred.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )
                pred_std = self.merge_patches_efficient(
                    pred_class.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )

                time_list.append((time.time() - time0))
                # print(pred_mean.shape)
                pred_mean = pred_mean.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                pred_std = pred_std.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                labels = labels.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                valid_mask = valid_mask.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                valid_mask = (valid_mask == self.args.forecast_inte).int()

                # 新建一个ex_name
                os.makedirs(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name), exist_ok=True)
                np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                                     f'pred_{step}.npy'), torch.cat([pred_mean, pred_std], 2).cpu().numpy())
                np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                                     f'label_{step}.npy'),
                        labels[:, :, 0, self.args.border_tar[0] + self.args.lat_dismiss[0]
                                        :pred_mean.shape[-2] + self.args.border_tar[0] + self.args.lat_dismiss[0],
                        self.args.border_tar[1] + self.args.lon_dismiss[0]
                        :pred_mean.shape[-1] + self.args.border_tar[1] + self.args.lon_dismiss[
                            0]].clone().cpu().numpy())

                all_means.append(pred_mean[:, :, 0].clone().cpu())
                all_stds.append(pred_std[:, :, 0].clone().cpu())
                # os.makedirs(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name), exist_ok=True)
                # np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                #                      f'pred_{step}.npy'), torch.cat([pred_mean, pred_std], 2).cpu().numpy())
                # np.save(os.path.join(self.results_dir, 'quali_figures', self.args.ex_name,
                #                      f'label_{step}.npy'),
                #         labels[:, :, 0, self.args.border_tar[0] + self.args.lat_dismiss[0]
                #                         :pred_mean.shape[-2] + self.args.border_tar[0] + self.args.lat_dismiss[0],
                #         self.args.border_tar[1] + self.args.lon_dismiss[0]
                #         :pred_mean.shape[-1] + self.args.border_tar[1] + self.args.lon_dismiss[
                #             0]].clone().cpu().numpy())

                all_labels.append(labels[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())
                all_masks.append(valid_mask[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())

                print(pred_mean.shape)
                print(pred_std.shape)
                print(labels.shape)
                print(valid_mask.shape)

                if self.args.empty_cache:
                    torch.cuda.empty_cache()

                CRPS_res = compute_crps_fully_vectorized(
                    pred_mean.cpu(), pred_std.cpu(),
                    labels[:, :, :, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].cpu(),
                    valid_mask[:, :, :, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].cpu()
                )
                CRPS_res_list.append(CRPS_res)
                del labels, pred_mean, pred_std, valid_mask
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # 合并所有batch
            all_means = torch.cat(all_means, dim=0)
            all_stds = torch.cat(all_stds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            # 展平空间维度
            N, T, H, W = all_means.shape
            print(all_means.shape)
            all_means_flat = all_means.permute(0, 2, 3, 1).flatten(0, 2)  # [N*H*W, T]
            all_stds_flat = all_stds.permute(0, 2, 3, 1).flatten(0, 2)
            all_labels_flat = all_labels.permute(0, 2, 3, 1).flatten(0, 2)
            all_masks_flat = all_masks.permute(0, 2, 3, 1).flatten(0, 2)  # [N*H*W, T]
            print(all_means_flat.shape)

            # 计算CSI指标
            # 计算CSI, POD, FAR指标
            csi_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds_6hour_accum)])
            pod_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds_6hour_accum)])
            far_results = np.zeros([self.args.aft_seq_length_test, len(self.precip_thresholds_6hour_accum)])

            for i, pt in enumerate(self.precip_thresholds_6hour_accum):
                # 计算 P(rain >= pt)
                probs = self._compute_prob_exceed_threshold(all_means_flat, all_stds_flat, pt)
                labels_binary = (all_labels_flat >= pt).float()

                for t in range(int(self.lead_times//self.args.forecast_inte)):
                    prob_thresh = self.optimal_prob_thresholds[t, i]
                    current_probs = probs[:, t]
                    current_labels = labels_binary[:, t]
                    current_mask = all_masks_flat[:, t]

                    csi = self._compute_csi_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )
                    pod = self._compute_pod_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )
                    far = self._compute_far_from_probs(
                        current_probs,
                        current_labels,
                        prob_thresh,
                        mask=current_mask
                    )

                    csi_results[t, i] = csi
                    pod_results[t, i] = pod
                    far_results[t, i] = far

                    if 1:
                        valid_count = (current_mask == 1).sum().item()
                        print(f"Test: lead_time={t + 1}, precip_thresh={pt}mm/h, "
                              f"prob_thresh={prob_thresh:.3f}, "
                              f"CSI={csi:.4f}, POD={pod:.4f}, FAR={far:.4f}, "
                              f"valid_samples={valid_count}")


            valid_crps = torch.stack(CRPS_res_list).nanmean(0).numpy()

            print('mean time:', np.mean(time_list))
            if epoch is not None:
                pkl_path = os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                        'metric_epoch' + str(epoch) + '.pkl')
            else:
                i = 0
                while os.path.exists(os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                                  'metric_' + str(test_epoch) + '_6hour_accum_' + str(i) + '.pkl')):
                    i += 1
                pkl_path = os.path.join(self.results_dir, 'csv_results', self.args.ex_name,
                                        'metric_' + str(test_epoch) + '_6hour_accum_' + str(i) + '.pkl')
            with open(pkl_path, 'wb') as file:
                pickle.dump(
                    [pod_results, far_results, csi_results, valid_crps], file)

    def optimize_on_validation_6hour_accum(self, val_dataloader, device, args,
                               search_resolution=50, mode='val', verbose=True):
        """
        在验证集上优化概率阈值
        使用PyTorch在CPU上计算

        Args:
            val_dataloader: 验证集数据加载器
            device: 设备（用于模型推理）
            args: 参数配置
            search_resolution: 概率阈值搜索粒度 (0到1之间搜索search_resolution个点)
            verbose: 是否打印进度

        Returns:
            optimal_thresholds: 最优概率阈值 [lead_times, len(precip_thresholds)]
        """
        self.model.eval()
        forcast_len = args.aft_seq_length_test

        # 收集所有预测和标签
        print("Collecting predictions and labels on validation set...")
        all_means = []
        all_stds = []
        all_labels = []  # 原始降水率标签
        all_masks = []
        time_list = []

        with torch.no_grad():
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data, rqi_data) in enumerate(
                    tqdm(val_dataloader, disable=not verbose)):
                # 数据预处理（保持你原有的代码不变）
                bs, bs_p, _, _, _, _ = L1_data.shape
                # print(L1_data.shape)
                L1_data = L1_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0, 1).type(torch.float32).to(self.device,
                                                                           non_blocking=True)
                rqi_data = rqi_data[:, :, None, ...].type(torch.float32).to(self.device, non_blocking=True)

                era5_data = torch.nan_to_num(era5_data, nan=0.0)
                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_mask_full = 1 - (torch.isnan(qpe_data_full) | (qpe_data_full < 0) | (qpe_data_full > self.args.max_qpe)).float()
                valid_mask = qpe_mask_full[:, self.args.in_len_val:].type(torch.int)
                valid_mask = valid_mask * ((rqi_data > self.args.rqi_thre).int())

                qpe_data_full = torch.nan_to_num(qpe_data_full, nan=0.0).clamp(min=0.0)
                qpe_data_full[qpe_data_full > self.args.max_qpe] = 0.0
                labels = qpe_data_full[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()

                qpe_data = torch.nan_to_num(qpe_data, nan=0.0).clamp_(min=0.0)
                qpe_data[qpe_data > self.args.max_qpe] = 0.0
                # labels = qpe_data[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()
                qpe_data = (qpe_data - self.args.max_min_qpe[1]) / (self.args.max_min_qpe[0] - self.args.max_min_qpe[1])

                L1_data = torch.nan_to_num(L1_data, nan=0.0).clamp_(min=0.0)
                L1_data.sub_(self.args.max_min[1])
                L1_data.div_(self.args.max_min[0] - self.args.max_min[1])

                inputs = torch.cat([L1_data, qpe_data[:, :self.args.in_len_val,
                                             ...]], dim=2)  # .type(torch.float32).to(self.device, non_blocking=True)

                time_data = time_data.type(torch.float32).repeat(inputs.shape[0], 1, 1).to(self.device,
                                                                                           non_blocking=True)

                geo_coords = torch.stack([self.geo_coords[:,
                                          lat_idx:lat_idx + self.coords_seg_size[0],
                                          lon_idx:lon_idx + self.coords_seg_size[1]] for
                                          (lat_idx, lon_idx) in self.test_indices]).to(non_blocking=True)
                dem_data = torch.stack([self.dem_data[:, int(
                    self.args.dem_ratio[0] * (lat_idx)):int(
                    self.args.dem_ratio[0] * (lat_idx + self.coords_seg_size[0])),
                                        int(self.args.dem_ratio[1] * (lon_idx)):int(
                                            self.args.dem_ratio[1] * (
                                                    lon_idx + self.coords_seg_size[1]))]
                                        for (lat_idx, lon_idx) in self.test_indices]).to(
                    non_blocking=True)

                time0 = time.time()

                chunk_size = 1
                pred, pred_class = torch.empty(0, forcast_len, *qpe_data.shape[-3:]) \
                    , torch.empty(0, forcast_len, *qpe_data.shape[-3:])  # .to(self.device)
                for i in range(int(inputs.shape[0] / chunk_size)):
                    # print(i)
                    pred_tmp, pred_std_tmp = self.model(inputs[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        era5_data[i * chunk_size:(i + 1) * chunk_size].clone()
                                                        , self.const_data,
                                                        time_data[i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        geo_coords=geo_coords[
                                                                   i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        dem_data=dem_data[
                                                                 i * chunk_size:(i + 1) * chunk_size].clone(),
                                                        # labels_qpe=labels,
                                                        aft_seq_length=forcast_len,
                                                        shrink=self.args.shrink, mode=mode,
                                                        device=self.device)
                    pred = torch.cat([pred, pred_tmp.cpu()])
                    pred_class = torch.cat([pred_class, pred_std_tmp.cpu()])

                pred_mean = self.merge_patches_efficient(
                    pred.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )
                pred_std = self.merge_patches_efficient(
                    pred_class.unflatten(0, [bs, bs_p]),
                    [self.max_indices[0] + self.args.tar_size[0] - 2 * self.args.border_tar[0],
                     self.max_indices[1] + self.args.tar_size[1] - 2 * self.args.border_tar[1]]
                )

                time_list.append((time.time() - time0))

                pred_mean = pred_mean.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                pred_std = pred_std.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                labels = labels.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                valid_mask = valid_mask.unflatten(1, [-1, self.args.forecast_inte]).sum(2)
                valid_mask = (valid_mask == self.args.forecast_inte).int()

                all_means.append(pred_mean[:, :, 0].clone().cpu())
                all_stds.append(pred_std[:, :, 0].clone().cpu())
                all_labels.append(labels[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                  :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())
                all_masks.append(valid_mask[:, :, 0, self.args.border_tar[0]+self.args.lat_dismiss[0]
                                                     :pred_mean.shape[-2] + self.args.border_tar[0]+self.args.lat_dismiss[0],
                                     self.args.border_tar[1]+self.args.lon_dismiss[0]
                                     :pred_mean.shape[-1] + self.args.border_tar[1]+self.args.lon_dismiss[0]].clone().cpu())

        # 合并所有batch（已在CPU上）
        all_means = torch.cat(all_means, dim=0)  # [N, T, H, W]
        all_stds = torch.cat(all_stds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)  # [N, T, H, W]

        # 展平空间维度
        N, T, H, W = all_means.shape
        print(all_means.shape)
        all_means_flat = all_means.permute(0,2,3,1).flatten(0, 2)  # [N*H*W, T]
        all_stds_flat = all_stds.permute(0,2,3,1).flatten(0, 2)
        all_labels_flat = all_labels.permute(0,2,3,1).flatten(0, 2)
        all_masks_flat = all_masks.permute(0,2,3,1).flatten(0, 2)   # [N*H*W, T]
        print(all_means_flat.shape)

        # 为每个降水强度阈值计算概率和标签
        print("Computing probabilities for each precipitation threshold...")
        all_probs_dict = {}
        all_labels_dict = {}
        all_masks_dict = {}  # 添加mask字典

        for pt in self.precip_thresholds_6hour_accum:
            # 计算 P(rain >= pt) (返回PyTorch Tensor)
            probs = self._compute_prob_exceed_threshold(all_means_flat, all_stds_flat, pt)
            all_probs_dict[pt] = probs  # [N*H*W, T]

            # 生成真实标签 (PyTorch Tensor)
            labels_binary = (all_labels_flat >= pt).float()
            all_labels_dict[pt] = labels_binary

            # mask保持不变（所有强度阈值共用同一个mask）
            all_masks_dict[pt] = all_masks_flat  # [N*H*W, T]

        # 搜索最优概率阈值
        print(f"Searching optimal probability thresholds (resolution={search_resolution})...")

        # 初始化最优阈值存储
        optimal_thresholds = np.zeros((self.lead_times, len(self.precip_thresholds_6hour_accum)))

        # 对每个降水强度阈值和每个预测时刻分别优化
        for i, pt in enumerate(self.precip_thresholds_6hour_accum):
            probs = all_probs_dict[pt]  # [N_samples, T]
            labels = all_labels_dict[pt]
            masks = all_masks_dict[pt]  # [N_samples, T]

            for t in range(int(self.lead_times//self.args.forecast_inte)):
                # 候选概率阈值
                if i < 5:
                    candidate_thresholds = np.linspace(0.01, 0.99, search_resolution)
                elif i == 5:
                    candidate_thresholds = np.linspace(0.002, 0.2, search_resolution)
                best_csi = 0.0
                best_thresh = 0.5  # 默认阈值

                # 获取当前时刻的mask
                current_mask = masks[:, t]  # [N_samples]

                # 检查是否有有效样本
                if (current_mask == 1).sum().item() == 0:
                    if verbose:
                        print(f"Warning: No valid samples for pt={pt}, t={t}, using default threshold=0.5")
                    optimal_thresholds[t, i] = 0.5
                    continue

                # 获取当前时刻的概率和标签
                current_probs = probs[:, t]  # [N_samples]
                current_labels = labels[:, t]  # [N_samples]

                # 搜索最优阈值
                for thresh in candidate_thresholds:
                    csi = self._compute_csi_from_probs(
                        current_probs,
                        current_labels,
                        thresh,
                        mask=current_mask
                    )
                    if csi > best_csi:
                        best_csi = csi
                        best_thresh = thresh

                optimal_thresholds[t, i] = best_thresh

                if verbose:
                    valid_count = (current_mask == 1).sum().item()
                    print(f"Precip threshold={pt}mm/h, lead_time={t + 1}: "
                          f"best_thresh={best_thresh:.3f}, best_CSI={best_csi:.4f}, "
                          f"valid_samples={valid_count}")

        self.optimal_prob_thresholds = optimal_thresholds

        return optimal_thresholds

    def get_summary_csi(self, csi_results):
        """
        汇总CSI结果，返回平均CSI和按阈值/时刻的统计
        """
        # 按降水强度阈值平均
        csi_by_threshold = {}
        for pt in self.precip_thresholds:
            csis = [csi for (t, p), csi in csi_results.items() if p == pt]
            csi_by_threshold[pt] = np.mean(csis)

        # 按预测时刻平均
        csi_by_leadtime = {}
        for t in range(self.lead_times):
            csis = [csi for (tt, p), csi in csi_results.items() if tt == t]
            csi_by_leadtime[t] = np.mean(csis)

        # 总体平均
        overall_csi = np.mean(list(csi_results.values()))

        return {
            'by_threshold': csi_by_threshold,
            'by_leadtime': csi_by_leadtime,
            'overall': overall_csi,
            'all_results': csi_results
        }
