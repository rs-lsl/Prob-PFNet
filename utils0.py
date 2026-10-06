# -*- coding: utf-8 -*-
import datetime
import os
import shutil
import sys


def get_days_in_year(start_year, end_year):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 1, 1)
    days = (end_date - start_date).days
    return days

def compute_crps_fully_vectorized(mean, std, target, mask, n_bins=512, bin_size=0.2):
    """
    修正版：保证不同 batch size 结果一致
    """
    bs, nt, c, h, w = mean.shape

    # 显式对齐 mask 维度，避免隐式广播不确定性
    if mask.dim() == 4 and c == 1:
        mask = mask.unsqueeze(2)  # [bs, nt, 1, h, w]

    # bins 与常数必须和输入同设备、同 dtype
    device = mean.device
    dtype = mean.dtype
    bins = torch.arange(0, n_bins + 1, device=device, dtype=dtype) * bin_size
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))

    valid_mask = (mask > 0)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [n_valid, 5]

    # ========== 关键修正 1：空样本返回 NaN 而不是 0 ==========
    if len(valid_indices) == 0:
        return torch.full((nt,), float('nan'), device=device, dtype=dtype)

    # 提取有效值
    mean_valid = mean[valid_mask].unsqueeze(1)      # [n_valid, 1]
    std_valid = std[valid_mask].unsqueeze(1)          # [n_valid, 1]
    target_valid = target[valid_mask].unsqueeze(1)  # [n_valid, 1]

    # 预测 CDF（正态分布）
    z = (bins.unsqueeze(0) - mean_valid) / (std_valid.clamp(min=1e-8))
    pred_cdf = 0.5 * (1 + torch.erf(z / sqrt2))

    # 真实 CDF（阶跃函数）
    true_cdf = (target_valid <= bins.unsqueeze(0)).float()

    # 每个像素的 CRPS
    crps_per_sample = ((pred_cdf - true_cdf) ** 2).sum(dim=1) * bin_size  # [n_valid]

    # 按时刻聚合
    time_indices = valid_indices[:, 1]  # 取时刻维度

    crps_per_time = torch.zeros(nt, device=device, dtype=dtype)
    count_per_time = torch.zeros(nt, device=device, dtype=dtype)

    crps_per_time.scatter_add_(0, time_indices, crps_per_sample)
    count_per_time.scatter_add_(0, time_indices, torch.ones_like(crps_per_sample))

    # ========== 关键修正 2：除 0 处置为 NaN，而非 0 ==========
    result = crps_per_time / count_per_time
    result[count_per_time == 0] = float('nan')
    return result

def create_folder_if_not_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def sort_by_last_digit(s):
    # 从字符串中提取最后的数字并作为排序依据
    return int(s.split('/')[-1])


def copy_all_files(source_folder, destination_folder):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    # 复制整个目录结构
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


def save_command():
    # 获取当前时间
    now = datetime.datetime.now()
    # 获取命令行参数
    command = " ".join(sys.argv)
    # 指定保存命令的文件路径
    file_path = "/home/lisl/saved_command.txt"
    # 将命令写入文件
    with open(file_path, "a") as file:
        file.write("\n" + str(now) + "\n" + command)
    print(f"save command: {file_path}")