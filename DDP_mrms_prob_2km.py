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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.special import erfc
from matplotlib.cm import ScalarMappable
from scipy.ndimage import binary_dilation

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
        self.optimal_prob_thresholds = None

        self.dem_data = np.load(file_name_dem)
        self.dem_data = torch.from_numpy(
            (self.dem_data - np.min(self.dem_data)) / (np.max(self.dem_data) - np.min(self.dem_data))).type(
            torch.float32).to(self.device)[None, ...]

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
            self.mean_std_era5[1] = torch.from_numpy(np.maximum(self.mean_std_era5[1].cpu().numpy(),
                                                                1e-2)).type(
                                    torch.float32).to(self.device)

    def test(self, mode='val', test_epoch=500):

        state_dict = torch.load(
            os.path.join(self.cp_dir, "weight.pth"))
        if self.args.dist:
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

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
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)

                bs, bs_p, _, _, _, _ = L1_data.shape
                # print(L1_data.shape)
                L1_data = L1_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0, 1).type(torch.float32).to(self.device,
                                                                           non_blocking=True)

                era5_data = torch.nan_to_num(era5_data, nan=0.0)
                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_mask_full = 1 - (torch.isnan(qpe_data_full) | (qpe_data_full < 0) | (qpe_data_full > self.args.max_qpe)).float()
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

                os.makedirs(os.path.join(self.results_dir, 'case_plots', self.args.ex_name),
                            exist_ok=True)

                # label 的裁剪方式和原来保存 npy 时完全一致
                label_plot = labels[:, :, 0,
                             self.args.border_tar[0] + self.args.lat_dismiss[0]
                             :pred_mean.shape[-2] + self.args.border_tar[0] + self.args.lat_dismiss[0],
                             self.args.border_tar[1] + self.args.lon_dismiss[0]
                             :pred_mean.shape[-1] + self.args.border_tar[1] + self.args.lon_dismiss[0]
                             ].clone()

                # print(torch.max(pred_mean), torch.min(pred_mean))
                # print(torch.max(pred_std), torch.min(pred_std))
                # print(torch.max(label_plot), torch.min(label_plot))
                self.save_case_plots_3dprob_decision(
                    step,
                    pred_mean=pred_mean,  # [1, 24, 1, H, W]
                    pred_std=pred_std,  # [1, 24, 1, H, W]
                    labels=label_plot,  # [1, 24, 1, H, W]
                    optimal_prob_thresholds=self.optimal_prob_thresholds,  # TODO: 换成你实际计算/加载的变量，形状 [24, 6]
                    hrrr_mean=None,  # 没有 HRRR 预报就传 None
                    sample_idx=0,
                    time_indices=[23, 11, 5, 0],
                    display_hours=[24, 12, 6, 1],
                    thresholds=[0.2, 1, 2, 4, 8, 20],
                    results_dir=self.results_dir,
                    ex_name=self.args.ex_name,
                )

                if self.args.empty_cache:
                    torch.cuda.empty_cache()

                del labels, pred_mean, pred_std, valid_mask
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    def save_case_plots_3dprob_decision(
            self,
            step,
            pred_mean,  # [B, T, 1, H, W]，torch.Tensor 或 np.ndarray
            pred_std,  # [B, T, 1, H, W]
            labels,  # [B, T, 1, H, W]
            optimal_prob_thresholds,  # np.ndarray, [T, n_thresholds]
            hrrr_mean=None,  # [B, T, 1, H, W] 或 None
            sample_idx=0,
            time_indices=(23, 17, 11, 5),
            display_hours=(24, 18, 12, 6),
            thresholds=(0.2, 1, 2, 4, 8, 20),
            results_dir=None,
            ex_name=None,
    ):
        """对单个 case 直接绘制并保存 3d_prob 和 decision 两张图（不经 npy 文件）。"""
        if optimal_prob_thresholds is None:
            raise ValueError("必须提供 optimal_prob_thresholds（形状 [T, n_thresholds]）")

        def to_np(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        def select_sample(arr):
            arr = np.squeeze(arr)  # [B,T,1,H,W] -> [T,H,W]（B=1 时）
            if arr.ndim == 4:  # B>1 时 -> 取第 sample_idx 个样本
                arr = arr[sample_idx]
            return arr

        mean_s = select_sample(to_np(pred_mean))
        std_s = select_sample(to_np(pred_std))
        label_s = select_sample(to_np(labels))
        use_hrrr = hrrr_mean is not None
        hrrr_mean_s = select_sample(to_np(hrrr_mean)) if use_hrrr else None

        data_h, data_w = mean_s.shape[-2:]
        if use_hrrr and hrrr_mean_s.shape[-2:] != (data_h, data_w):
            raise ValueError(f'HRRR 空间尺寸 {hrrr_mean_s.shape[-2:]} 与 Ours {(data_h, data_w)} 不一致')

        save_dirs = {
            '3d_prob': os.path.join(results_dir, 'case_plots', ex_name, '3d_prob'),
            'decision': os.path.join(results_dir, 'case_plots', ex_name, 'decision'),
        }
        os.makedirs(save_dirs['3d_prob'], exist_ok=True)
        os.makedirs(save_dirs['decision'], exist_ok=True)

        dpi_all = 1000
        dpi_all_tile = 500

        lead_time_font = 18
        row_label_font = 18

        tile_h, tile_w = data_h, data_w
        sub_h, sub_w = 240, int(240 * 2.56)

        LAT_RANGE = (29.6, 44.4)
        LON_RANGE = (250.8, 285.2)
        display_extent = [LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]]

        time_hour_pairs = sorted(zip(time_indices, display_hours), key=lambda item: item[1])
        if len(time_hour_pairs) != 4:
            raise ValueError("time_indices 和 display_hours 必须各包含 4 个元素")
        time_groups = [time_hour_pairs[:2], time_hour_pairs[2:]]

        def gen_map_tile():
            dpi = dpi_all_tile
            fig_m = plt.figure(figsize=(tile_w / dpi, tile_h / dpi), dpi=dpi,
                               frameon=False, facecolor='white')
            ax_m = fig_m.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
            ax_m.set_extent(display_extent, crs=ccrs.PlateCarree())
            ax_m.patch.set_facecolor('white')
            ax_m.patch.set_alpha(1.0)
            ax_m.add_feature(cfeature.STATES.with_scale('110m'), linewidth=0.1,
                             edgecolor='#444444', alpha=0.9, facecolor='none')
            ax_m.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.1,
                             edgecolor='#444444', alpha=0.9)
            ax_m.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.1,
                             edgecolor='#444444', alpha=0.9)
            ax_m.set_axis_off()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=dpi_all_tile, pad_inches=0,
                        transparent=False, facecolor='white', edgecolor='none')
            plt.close()
            buf.seek(0)
            img = Image.open(buf).convert('RGBA')
            if img.size != (tile_w, tile_h):
                img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            arr = np.array(img)
            bg = np.full((tile_h, tile_w, 4), 255, dtype=np.uint8)
            bg[:, :, :3] = arr[:, :, :3]
            bg[:, :, 3] = 255
            return bg

        base_map = gen_map_tile()
        cmap_mrms, norm_mrms = create_colormap_mrms_qpe(50)
        cmap_prob = plt.cm.summer_r

        layer_colors = [
            [0.0, 0.6, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [1.0, 0.6, 0.0], [1.0, 0.0, 0.0], [0.8, 0.4, 0.8]
        ]
        layer_colors_dark = [
            tuple(np.clip(np.asarray(color) * 0.70, 0, 1))
            for color in layer_colors
        ]

        def add_threshold_probability_legend(fig, rect=None):
            from matplotlib.patches import Rectangle
            if rect is None:
                rect = [0.15, 0.020, 0.70, 0.030]
            legend_ax = fig.add_axes(rect)
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(-0.65, 1.35)
            legend_ax.axis('off')
            n_threshold = len(thresholds)
            gap = 0.012
            segment_w = (1 - (n_threshold - 1) * gap) / n_threshold
            bar_y0 = 0.20
            bar_h = 0.60
            gradient = np.linspace(0, 1, 256)[None, :]
            for i, (pt, color) in enumerate(zip(thresholds, layer_colors_dark)):
                x0 = i * (segment_w + gap)
                x1 = x0 + segment_w
                xc = (x0 + x1) / 2
                cmap_i = mcolors.LinearSegmentedColormap.from_list(
                    f'threshold_legend_{i}', ['#FFFFFF', color])
                legend_ax.imshow(gradient, extent=[x0, x1, bar_y0, bar_y0 + bar_h],
                                 aspect='auto', cmap=cmap_i, vmin=0, vmax=1,
                                 interpolation='bicubic')
                legend_ax.add_patch(Rectangle((x0, bar_y0), segment_w, bar_h,
                                              fill=False, edgecolor='black', linewidth=0.8))
                legend_ax.text(xc, 1.02, '0%–100%', ha='center', va='bottom',
                               fontsize=10, color='black', clip_on=False)
                legend_ax.text(xc, -0.15, f'≥{pt:g} mm', ha='center', va='top',
                               fontsize=10, color='black', clip_on=False)

        def fig_to_rgba(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi_all, pad_inches=0, transparent=True)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert('RGBA')
            if img.size != (tile_w, tile_h):
                img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            return np.array(img)

        def render_observation_precip_tile(l2d):
            l2d_display = np.where(l2d >= 0.2, l2d, np.nan)
            fig_sub = plt.figure(figsize=(tile_w / 200, tile_h / 200), dpi=dpi_all, frameon=False)
            ax_sub = fig_sub.add_axes([0, 0, 1, 1])
            ax_sub.set_xlim(0, tile_w)
            ax_sub.set_ylim(0, tile_h)
            ax_sub.axis('off')
            ax_sub.imshow(base_map, extent=[0, tile_w, 0, tile_h])
            ax_sub.imshow(l2d_display, cmap=cmap_mrms, norm=norm_mrms,
                          extent=[0, tile_w, 0, tile_h])
            return fig_to_rgba(fig_sub)

        def render_decision_tile(m2d, s2d=None, t=None, deterministic=False):
            fig_sub = plt.figure(figsize=(tile_w / 200, tile_h / 200), dpi=dpi_all, frameon=False)
            ax_sub = fig_sub.add_axes([0, 0, 1, 1])
            ax_sub.set_xlim(0, tile_w)
            ax_sub.set_ylim(0, tile_h)
            ax_sub.axis('off')
            ax_sub.imshow(base_map, extent=[0, tile_w, 0, tile_h])
            for i, pt in enumerate(thresholds):
                if deterministic:
                    valid = m2d >= pt
                else:
                    pr = prob_exceed(m2d, s2d, pt)
                    opt_th = optimal_prob_thresholds[t, i]
                    valid = pr > opt_th
                if not valid.any():
                    continue
                rgba = np.full((tile_h, tile_w, 4), 255, dtype=np.uint8)
                rgba[:, :, :3] = (np.array(layer_colors[i]) * 255).astype(np.uint8)
                rgba[:, :, 3] = 0
                rgba[valid, 3] = 255
                ax_sub.imshow(rgba, extent=[0, tile_w, 0, tile_h])
            return fig_to_rgba(fig_sub)

        def paste_tile(full_img, tile, x, y):
            tile_resized = resize_tile(tile, sub_h, sub_w)
            alpha = tile_resized[:, :, 3:4] / 255.0
            region = full_img[y:y + sub_h, x:x + sub_w]
            region[:, :, :3] = (
                    region[:, :, :3] * (1 - alpha) + tile_resized[:, :, :3] * alpha
            ).astype(np.uint8)
            region[:, :, 3] = 255

        def add_grouped_labels(ax, row_labels, group_h, top_m, group_gap, row_gap):
            for group_idx, group in enumerate(time_groups):
                group_y = top_m + group_idx * (group_h + group_gap)
                for col, (_, hour) in enumerate(group):
                    xc = left_m + col * (sub_w + col_gap) + sub_w // 2
                    ax.text(xc, group_y, f'+{hour}h', ha='center', va='bottom',
                            fontsize=lead_time_font, fontweight='bold', color='red')
                for row, label in enumerate(row_labels):
                    yc = group_y + row * (sub_h + row_gap) + sub_h // 2
                    ax.text(25, yc, label, ha='center', va='center',
                            fontsize=row_label_font, color='black',
                            rotation=90, rotation_mode='anchor')

        def add_csi_table_below(ax, x, y_bottom, csi_values, thr_list, bold_mask=None):
            label_w = 70
            n = len(thr_list)
            col_w = (sub_w - label_w) / n
            y_thr = y_bottom + 8
            y_csi = y_bottom + 40
            ax.add_patch(plt.Rectangle(
                (x + label_w, y_thr - 2), sub_w - label_w, 28,
                facecolor='#EDEDED', edgecolor='none', zorder=0))
            ax.text(x + 6, y_csi, 'CSI', ha='left', va='top',
                    fontsize=13, fontweight='bold', color='black')
            for i, (thr, csi) in enumerate(zip(thr_list, csi_values)):
                xc = x + label_w + i * col_w + col_w / 2
                ax.text(xc, y_thr, f'{thr:g}', ha='center', va='top',
                        fontsize=12, color='black')
                fw = 'bold' if (bold_mask is not None and bold_mask[i]) else 'normal'
                ax.text(xc, y_csi, f'{csi:.2f}', ha='center', va='top',
                        fontsize=12, fontweight=fw, color='black')

        def add_lat_lon_annotations(ax, n_rows, group_h, top_m, gg, rg):
            lat_ticks = [30, 35, 40]
            lon_ticks_w = [100, 80]
            lon_ticks_raw = [260, 280]
            for group_idx, group in enumerate(time_groups):
                group_y_pos = top_m + group_idx * (group_h + gg)
                for row in range(n_rows):
                    y_tile = group_y_pos + row * (sub_h + rg)
                    x_text = left_m + 10
                    for lat in lat_ticks:
                        ratio = (44.4 - lat) / (44.4 - 29.6)
                        y_text = y_tile + ratio * sub_h
                        ax.text(x_text, y_text, f'{lat}°N', ha='right', va='center',
                                fontsize=LAT_LON_FONT, color='black',
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                          edgecolor='none', alpha=0.8))
            for group_idx, group in enumerate(time_groups):
                group_y_pos = top_m + group_idx * (group_h + gg)
                y_tile_bottom = group_y_pos + (n_rows - 1) * (sub_h + rg) + sub_h
                y_text = y_tile_bottom
                for col, _ in enumerate(group):
                    x_tile = left_m + col * (sub_w + col_gap)
                    for lon_raw, lon_w in zip(lon_ticks_raw, lon_ticks_w):
                        ratio = (lon_raw - 250.8) / (285.2 - 250.8)
                        x_text = x_tile + ratio * sub_w
                        ax.text(x_text, y_text, f'{lon_w}°W', ha='center', va='bottom',
                                fontsize=LAT_LON_FONT, color='black',
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                          edgecolor='none', alpha=0.8))

        row_gap, col_gap = 20, 20
        left_m, top_m = 110, 60
        bottom_m_3d_plot = 330
        bottom_m_decision = 250
        border_w = 3
        row_gap_csi = 95
        group_gap_csi = 170
        LAT_LON_FONT = 11

        # ================== 3D-like 多层概率轮廓（每个 case 1 张） ==================
        row_labels = ['Observation'] + (['HRRR'] if use_hrrr else []) + ['Ours']
        n_method_rows = len(row_labels)
        row_hrrr = 1 if use_hrrr else None  # HRRR 所在行号（没有则为 None）
        row_ours = n_method_rows - 1  # Ours 永远在最后一行
        rg, gg = row_gap_csi, group_gap_csi
        group_h = n_method_rows * sub_h + (n_method_rows - 1) * rg
        total_w = left_m + 2 * sub_w + col_gap
        total_h = top_m + 2 * group_h + gg + bottom_m_3d_plot
        full_img = np.full((total_h, total_w, 4), 255, dtype=np.uint8)
        csi_annotations = []

        for group_idx, group in enumerate(time_groups):
            group_y = top_m + group_idx * (group_h + gg)
            for col, (t, _) in enumerate(group):
                x = left_m + col * (sub_w + col_gap)
                m2d = mean_s[t]
                s2d = std_s[t]
                l2d = label_s[t]

                # 第一行：真值
                y_obs = group_y
                paste_tile(full_img, render_observation_precip_tile(l2d), x, y_obs)
                add_border(full_img, x, y_obs, sub_h, sub_w, border_w)

                # 中间行：HRRR（仅在有 HRRR 时绘制）
                hrrr_csis = None
                y_hrrr = None
                if use_hrrr:
                    h2d = hrrr_mean_s[t]
                    y_hrrr = group_y + row_hrrr * (sub_h + rg)
                    paste_tile(full_img, render_decision_tile(h2d, deterministic=True),
                               x, y_hrrr)
                    add_border(full_img, x, y_hrrr, sub_h, sub_w, border_w)
                    hrrr_csis = [compute_csi(h2d >= pt, l2d >= pt) for pt in thresholds]

                # 最后一行：Ours 3D-like 概率轮廓
                fig_sub = plt.figure(figsize=(tile_w / 200, tile_h / 200), dpi=dpi_all,
                                     frameon=False)
                ax_sub = fig_sub.add_axes([0, 0, 1, 1])
                ax_sub.set_xlim(0, tile_w)
                ax_sub.set_ylim(0, tile_h)
                ax_sub.axis('off')
                ax_sub.imshow(base_map, extent=[0, tile_w, 0, tile_h])

                ours_csis = []
                for i, pt in enumerate(thresholds):
                    pr = prob_exceed(m2d, s2d, pt)
                    opt_th = optimal_prob_thresholds[t, i]
                    valid = pr > opt_th
                    ours_csis.append(compute_csi(valid, l2d >= pt))
                    if not valid.any():
                        continue

                    offset = 4
                    shadow = np.full_like(pr, np.nan)
                    shadow[offset:, offset:] = pr[:-offset, :-offset]
                    smask = valid & ~np.isnan(shadow) & (shadow > opt_th)
                    if smask.any():
                        rgba = np.full((tile_h, tile_w, 4), [60, 60, 60, 90], dtype=np.uint8)
                        rgba[~smask] = [255, 255, 255, 0]
                        ax_sub.imshow(rgba, extent=[0, tile_w, 0, tile_h])

                    base_c = layer_colors[i]
                    dark_c = layer_colors_dark[i]
                    cmap_layer = mcolors.LinearSegmentedColormap.from_list(
                        f'L{i}', ['#FFFFFF', dark_c])
                    disp = np.full_like(pr, np.nan)
                    disp[valid] = pr[valid]
                    Y, X = np.mgrid[0:tile_h, 0:tile_w]
                    disp_plot = disp[::-1]
                    ax_sub.contourf(X, Y, disp_plot,
                                    levels=np.linspace(opt_th, 1.0, 10),
                                    cmap=cmap_layer, alpha=0.7)
                    ax_sub.contour(X, Y, disp_plot, levels=[opt_th],
                                   colors=[base_c], linewidths=2.5)
                    ax_sub.contour(X, Y, disp_plot, levels=[opt_th],
                                   colors='black', linewidths=2)

                y_ours = group_y + row_ours * (sub_h + rg)
                paste_tile(full_img, fig_to_rgba(fig_sub), x, y_ours)
                add_border(full_img, x, y_ours, sub_h, sub_w, border_w)
                csi_annotations.append((x, y_hrrr, hrrr_csis, y_ours + sub_h, ours_csis))

        fig = plt.figure(figsize=(total_w / 150, total_h / 150), dpi=dpi_all)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(full_img, extent=[0, total_w, total_h, 0])
        ax.set_xlim(0, total_w)
        ax.set_ylim(total_h, 0)
        ax.axis('off')
        add_grouped_labels(ax, row_labels, group_h, top_m, gg, rg)

        for (xa, yb_h, h_csis, yb_o, o_csis) in csi_annotations:
            if use_hrrr:
                h_bold = [h > o for h, o in zip(h_csis, o_csis)]
                o_bold = [o > h for h, o in zip(h_csis, o_csis)]
                add_csi_table_below(ax, xa, yb_h, h_csis, thresholds, bold_mask=h_bold)
                add_csi_table_below(ax, xa, yb_o, o_csis, thresholds, bold_mask=o_bold)
            else:
                add_csi_table_below(ax, xa, yb_o, o_csis, thresholds)

        add_lat_lon_annotations(ax, n_method_rows, group_h, top_m, gg, rg)
        # ……下方 colorbar、渐变图例、savefig 部分保持原样不变……

        sm = ScalarMappable(norm=norm_mrms, cmap=cmap_mrms)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.15, 0.050, 0.70, 0.016])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([0, 0.2, 1, 2, 4, 8, 20, 50])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='x', labelsize=14, pad=2)
        cbar.set_label('Hourly Accumulated Precipitation Rate (mm/h)',
                       fontsize=14, labelpad=8)

        add_threshold_probability_legend(fig, rect=[0.15, 0.0, 0.70, 0.030])

        fname = f'case_3d_prob_step{step}_sample{sample_idx}'
        plt.savefig(os.path.join(save_dirs['3d_prob'], fname + '.png'), dpi=dpi_all,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f'Saved 3d_prob: {fname}.png')

        # ================== 真值 + HRRR + Ours 决策图层（每个 case 1 张） ==================
        row_labels = ['Observation'] + (['HRRR'] if use_hrrr else []) + ['ILFNet']
        n_method_rows = len(row_labels)
        row_hrrr = 1 if use_hrrr else None
        row_ours = n_method_rows - 1
        rg, gg = row_gap_csi, group_gap_csi
        group_h = n_method_rows * sub_h + (n_method_rows - 1) * rg
        total_w = left_m + 2 * sub_w + col_gap
        total_h = top_m + 2 * group_h + gg + bottom_m_decision
        full_img = np.full((total_h, total_w, 4), 255, dtype=np.uint8)
        csi_annotations = []

        for group_idx, group in enumerate(time_groups):
            group_y = top_m + group_idx * (group_h + gg)
            for col, (t, _) in enumerate(group):
                x = left_m + col * (sub_w + col_gap)
                m2d = mean_s[t]
                s2d = std_s[t]
                l2d = label_s[t]

                y_obs = group_y
                paste_tile(full_img, render_observation_precip_tile(l2d), x, y_obs)
                add_border(full_img, x, y_obs, sub_h, sub_w, border_w)

                hrrr_csis = None
                y_hrrr = None
                if use_hrrr:
                    h2d = hrrr_mean_s[t]
                    y_hrrr = group_y + row_hrrr * (sub_h + rg)
                    paste_tile(full_img, render_decision_tile(h2d, deterministic=True),
                               x, y_hrrr)
                    add_border(full_img, x, y_hrrr, sub_h, sub_w, border_w)
                    hrrr_csis = [compute_csi(h2d >= pt, l2d >= pt) for pt in thresholds]

                y_ours = group_y + row_ours * (sub_h + rg)
                paste_tile(full_img, render_decision_tile(m2d, s2d=s2d, t=t,
                                                          deterministic=False),
                           x, y_ours)
                add_border(full_img, x, y_ours, sub_h, sub_w, border_w)

                ours_csis = []
                for i, pt in enumerate(thresholds):
                    pr = prob_exceed(m2d, s2d, pt)
                    opt_th = optimal_prob_thresholds[t, i]
                    ours_csis.append(compute_csi(pr > opt_th, l2d >= pt))
                csi_annotations.append((x, y_hrrr, hrrr_csis, y_ours + sub_h, ours_csis))

        fig = plt.figure(figsize=(total_w / 150, total_h / 150), dpi=dpi_all)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(full_img, extent=[0, total_w, total_h, 0])
        ax.set_xlim(0, total_w)
        ax.set_ylim(total_h, 0)
        ax.axis('off')
        add_grouped_labels(ax, row_labels, group_h, top_m, gg, rg)

        for (xa, yb_h, h_csis, yb_o, o_csis) in csi_annotations:
            if use_hrrr:
                h_bold = [h > o for h, o in zip(h_csis, o_csis)]
                o_bold = [o > h for h, o in zip(h_csis, o_csis)]
                add_csi_table_below(ax, xa, yb_h, h_csis, thresholds, bold_mask=h_bold)
                add_csi_table_below(ax, xa, yb_o, o_csis, thresholds, bold_mask=o_bold)
            else:
                add_csi_table_below(ax, xa, yb_o, o_csis, thresholds)

        add_lat_lon_annotations(ax, n_method_rows, group_h, top_m, gg, rg)

        sm = ScalarMappable(norm=norm_mrms, cmap=cmap_mrms)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.15, 0.035, 0.7, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Hourly Accumulated Precipitation Rate (mm)', fontsize=14)
        cbar.set_ticks([0, 0.2, 1, 2, 4, 8, 20, 50])
        cbar.ax.tick_params(labelsize=14)

        fname = f'case_decision_step{step}_sample{sample_idx}'
        plt.savefig(os.path.join(save_dirs['decision'], fname + '.png'), dpi=dpi_all,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f'Saved decision: {fname}.png')

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

def create_colormap_mrms_qpe(range_max=50):
    colors = [
        [1, 1, 1], [0, 0.6, 1], [0, 1, 0], [1, 1, 0],
        [1, 0.6, 0], [1, 0, 0], [0.8, 0.4, 0.8]
    ]
    boundaries = [0, 0.2, 1, 2, 4, 8, 20, range_max]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def prob_exceed(mean, std, thr):
    """计算 P(X >= thr) 的高斯尾部概率"""
    z = (thr - mean) / (std + 1e-8)
    return 0.5 * erfc(z / np.sqrt(2))


def compute_csi(pred_bin, label_bin):
    """
    计算 CSI = TP / (TP + FP + FN)

    Args:
        pred_bin: 二元预测 (bool 或 0/1 数组)，True 表示判定发生
        label_bin: 二元真值 (bool 或 0/1 数组)，True 表示实际发生

    Returns:
        CSI 值 (float)；若 TP+FP+FN == 0 则返回 0.0
    """
    pred_bin = np.asarray(pred_bin).astype(bool)
    label_bin = np.asarray(label_bin).astype(bool)
    TP = np.logical_and(pred_bin, label_bin).sum()
    FP = np.logical_and(pred_bin, ~label_bin).sum()
    FN = np.logical_and(~pred_bin, label_bin).sum()
    denom = TP + FP + FN
    if denom == 0:
        return 0.0
    return float(TP) / float(denom)


def create_colormap_mrms_qpe(range_max=50):
    colors = [
        [1, 1, 1], [0, 0.6, 1], [0, 1, 0], [1, 1, 0],
        [1, 0.6, 0], [1, 0, 0], [0.8, 0.4, 0.8]
    ]
    boundaries = [0, 0.2, 1, 2, 4, 8, 20, range_max]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def resize_tile(tile_arr, target_h, target_w):
    """将 tile resize 为指定长方形尺寸"""
    img = Image.fromarray(tile_arr, mode='RGBA')
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)


def add_border(full_img, x, y, h, w, bw=3, color=(0, 0, 0, 255)):
    """在 full_img 的指定区域添加边框（仿照第二个代码 BORDER_WIDTH = 3，黑色）"""
    full_img[y:y + bw, x:x + w] = color
    full_img[y + h - bw:y + h, x:x + w] = color
    full_img[y:y + h, x:x + bw] = color
    full_img[y:y + h, x + w - bw:x + w] = color
