import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from modules.vit import VisionTransformer
from modules.multi_layers import SelfAttention_patch
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn


class UncertaintyAwareInput(nn.Module):
    def __init__(self, time_inte: int):
        super().__init__()
        self.time_inte = time_inte

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        nn.init.constant_(self.alpha, -0.4328)
        nn.init.constant_(self.beta, 0.5413)

    def forward(self, latent_mean, latent_std):
        a = torch.nn.functional.softplus(self.alpha) + 0.5 #
        b = torch.nn.functional.softplus(self.beta) + 1.0  #

        uncertainty_gate = torch.exp(-a * latent_std)
        modulated = latent_mean * uncertainty_gate * b

        return modulated

class SimVP_Model_x(nn.Module):
    def __init__(self, in_shape, out_ch, hid_S=32, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.1, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, args=None, device=None, **kwargs):
        super(SimVP_Model_x, self).__init__()
        B, T, C, H, W = in_shape  # T is input_time_length

        self.args = args
        self.hid_S = hid_S
        self.out_ch = out_ch

        self.UA_net = nn.ModuleList([UncertaintyAwareInput(ii) for ii in self.args.time_inte])

        self.H_d, self.W_d = args.H_d, args.W_d
        # Encoder, Decoder
        print('drop_rate: ', args.drop)
        act_inplace = False
        self.enc = nn.Sequential(
            Encoder(C, args.era5_shape[0], hid_S, N_S, spatio_kernel_enc, H, W, self.H_d, self.W_d,
                    act_inplace=act_inplace, drop_rate=args.drop, n_head=2, patch_size=4, args=args))       #   C_in, C_hid, N_S, spatio_kernel, act_inplace=True

        self.dec = Decoder(hid_S, out_ch, N_S, spatio_kernel_dec, self.H_d, self.W_d,
                           args.tar_size[0]-2*args.border_tar[0], args.tar_size[1]-2*args.border_tar[1],
                           act_inplace=act_inplace, device=device, args=args, patch_size=4) # patch_size鏈€濂藉ぇ浜庢渶缁堜笂閲囨牱姣斾緥  # 1 means the total_precipitation_6hr var, 瑙ｇ爜鍣ㄨ緭鍑虹殑g_theta_pred 鏂瑰樊瑕佸ぇ
        self.decoder_recon = Decoder_recon(hid_S, args.tar_era5_shape, args.tar_L1_shape, 2, spatio_kernel_dec, H=self.H_d, W=self.H_d,
                                           act_inplace=act_inplace, device=device, drop_rate=args.drop)

        self.time_embedding = nn.Sequential(
                nn.Linear(args.input_time_length*args.time_emb_num, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),   #  hid_S  ***
                nn.LeakyReLU(),
                nn.Linear(256, int(hid_S*0.2))
            )

        # prediction model
        scale_fac = args.input_time_length
        num_block = 12
        N_S_pred = 4
        norm_band_num = 4  # in self attention
        patch_size = 4     # in self attention
        step = 2

        patch_size_list = [4,4,4]
        drop_ratio_list = [1.0, 1.0, 1.0]
        # 鍏堥獙淇℃伅锛氭椂闂撮棿闅旇秺澶э紝棰勬祴鐨勬柟宸秺澶?
        self.hid = [nn.Sequential(  # out_chans鎸囪緭鍑虹殑棰勬祴鍒囧悜閲忥紙鍧囧€煎拰鏍囧噯宸級锛宔mbed_dim灏濊瘯璁剧疆涓簅ut_chans*ps*ps
            VisionTransformer([self.H_d, self.W_d], patch_size=[patch_size_list[i],patch_size_list[i]],
                              inp_chans=scale_fac*hid_S+int(hid_S*0.2),
                              out_chans=hid_S*2,
                                     embed_dim=768, depth=8, num_heads=12, mlp_ratio=4., qkv_bias=True,
                              mlp_drop_rate=args.drop*drop_ratio_list[i],
                                        attn_drop_rate=args.drop*drop_ratio_list[i],
                              path_drop_rate=args.drop*drop_ratio_list[i], norm_layer="layer_norm", comm_inp_name="fin",
                                     comm_hidden_name="fout")
        )
            for i in range(len(self.args.time_inte))]  #  args.drop
        self.hid = nn.ModuleList(self.hid)

    def forward(self, x_raw, era5_data_input, const_data, time_data, era5_data_label=None, labels=None, labels_qpe=None,
                geo_coords=None, dem_data=None, diff_ori=None,
                aft_seq_length=1, hid_i=0, shrink=1, mode='train', device=None, **kwargs):
        B, T, C, H, W = x_raw.shape
        era5_data = era5_data_input.flatten(0,1)
        x = x_raw.flatten(0,1)

        embed_in_mean, embed_in_std = checkpoint(self.enc,
                                                 [era5_data, x,
                                                  geo_coords.repeat(T, 1, 1, 1), dem_data.repeat(T, 1, 1, 1)], use_reentrant=False)

        _, C_, H_, W_ = embed_in_mean.shape

        z = embed_in_mean.unflatten(0, (B, T)).flatten(1, 2)  # .detach()
        z_std = embed_in_std.unflatten(0, (B, T))
        const_emb = None  # 0fas

        hid_mean, hid_std = self._predict_pangu2(z, z_std.flatten(1, 2), const_emb, time_data, aft_seq_length, shrink=shrink, mode=mode,
                                  device=device)
        Y_mean, Y_std = self.dec(hid_mean.unflatten(1, [aft_seq_length, -1]).flatten(0, 1),
                                 hid_std.unflatten(1, [aft_seq_length, -1]).flatten(0, 1), device=device)  # ,pred_class杈撳嚭 鍍忕礌涓烘棤闆ㄧ殑姒傜巼锛屽綋閫佸叆focal loss鏃跺姞绗﹀彿锛岀湡瀹炴爣绛捐緭鍏?-true_class


        return (Y_mean*(self.args.max_min_qpe[0]-self.args.max_min_qpe[1])+self.args.max_min_qpe[1]).unflatten(0,[B, aft_seq_length])\
            , (Y_std*(self.args.max_min_qpe[0]-self.args.max_min_qpe[1])).unflatten(0,[B, aft_seq_length])#(Y_mean*(self.args.max_min_qpe[0]-self.args.max_min_qpe[1])+self.args.max_min_qpe[1]).unflatten(0,[B, aft_seq_length])\

    def exp_map(self, p: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        B, twoC, H, W = p.shape
        C = twoC // 2

        mu, sigma = p[:, :C], p[:, C:]
        sigma = torch.clamp(sigma, min=self.args.eps)

        m, s = tangent[:, :C], tangent[:, C:]

        s = s.clamp(-5.0, 5.0)
        sigma_next = (sigma * torch.exp(s)).clamp(max=10.0)
        mu_next = mu + m * sigma


        return mu_next, sigma_next


    def _predict_pangu2(self, cur_seq, cur_seq_std, cur_const_data, cur_time_data, aft_seq_length, shrink=False, mode='val',
                        device=None,
                        **kwargs):
        in_len = self.args.input_time_length
        pred_y = cur_seq.clone()
        pred_y_std = cur_seq_std.clone()
        pred_mean_list = []
        pred_log_var_list = []
        # re_time_inte = self.args.time_inte[::-1]
        max_inte = max(self.args.time_inte)
        time_inte_dict = dict(zip(self.args.time_inte, np.arange(len(self.args.time_inte))))

        for pred_i in range(1, aft_seq_length + 1):
            # print(pred_i)

            iter_num = []  # iter_num[2,1,1]--inte:[4,2,1]

            pred_pos = pred_i % max_inte
            if pred_i == 1 or pred_i == 2:
                # print(pred_pos)
                # print(pred_i)
                temp_pred_y = [pred_y[:, -(pred_pos*i)*self.hid_S:-(pred_pos*i-1)*self.hid_S] for i in range(in_len, 0, -1)] if pred_i == 2 else \
                                [pred_y[:, -(pred_pos * i) * self.hid_S:-(pred_pos * i - 1) * self.hid_S] for i in range(in_len, 1, -1)] + \
                                [pred_y[:, -(pred_pos * 1) * self.hid_S:]]
                temp_pred_y_std = [pred_y_std[:, -(pred_pos*i)*self.hid_S:-(pred_pos*i-1)*self.hid_S] for i in range(in_len, 0, -1)] if pred_i == 2 else \
                                [pred_y_std[:, -(pred_pos * i) * self.hid_S:-(pred_pos * i - 1) * self.hid_S] for i in range(in_len, 1, -1)] + \
                                [pred_y_std[:, -(pred_pos * 1) * self.hid_S:]]

                time_emb_idx = [self.args.in_len_val - ii*pred_pos for ii in range(in_len, 0, -1)] if pred_i == 1 else \
                                [self.args.in_len_val - ii * pred_pos + 1 for ii in range(in_len, 0, -1)]
                temp_out = self.forward_recur(self.UA_net[time_inte_dict[pred_pos]](torch.cat(temp_pred_y, 1), torch.cat(temp_pred_y_std, 1)),
                                                 cur_const_data, cur_time_data[:, :,
                                                                 time_emb_idx]
                                                 , device, hid_i=time_inte_dict[pred_pos])
                if pred_pos == 1:
                    pred_mean, cur_seq_std = self.exp_map(torch.cat([pred_y[:, -self.hid_S:],
                                                                 pred_y_std[:, -self.hid_S:]], 1),
                                                  temp_out)
                else:
                    pred_mean, cur_seq_std = self.exp_map(torch.cat([pred_y[:, -pred_pos*self.hid_S:-(pred_pos-1)*self.hid_S],
                                                                 pred_y_std[:, -pred_pos*self.hid_S:-(pred_pos-1)*self.hid_S]], 1),
                                                  temp_out)

            elif pred_i == 3:
                time_emb_idx = [self.args.in_len_val - ii * 2 + 2 for ii in range(in_len, 0, -1)]
                temp_out = self.forward_recur(self.UA_net[time_inte_dict[2]](torch.cat([pred_y[:, -(2*i)*self.hid_S:-(2*i-1)*self.hid_S] for i in range(in_len, 0, -1)], 1),
                                                                            torch.cat([pred_y_std[:, -(2*i)*self.hid_S:-(2*i-1)*self.hid_S] for i in range(in_len, 0, -1)], 1)),
                                                 cur_const_data, cur_time_data[:, :,
                                                                 time_emb_idx]
                                                 , device, hid_i=time_inte_dict[2])
                time_inteval = 2
                pred_mean, cur_seq_std = self.exp_map(torch.cat([pred_y[:, -time_inteval*self.hid_S:-(time_inteval-1)*self.hid_S],
                                                                 pred_y_std[:, -time_inteval*self.hid_S:-(time_inteval-1)*self.hid_S]], 1),
                                                  temp_out)
            else:
                time_emb_idx = [self.args.in_len_val + pred_i - ii * 4 + -1 for ii in range(in_len, 0, -1)]
                temp_out = self.forward_recur(self.UA_net[time_inte_dict[4]](torch.cat([pred_y[:, -(4*i)*self.hid_S:-(4*i-1)*self.hid_S] for i in range(in_len, 0, -1)], 1),
                                                                             torch.cat([pred_y_std[:, -(4*i)*self.hid_S:-(4*i-1)*self.hid_S] for i in range(in_len, 0, -1)], 1)),
                                                 cur_const_data, cur_time_data[:, :,
                                                                 time_emb_idx]
                                                 , device, hid_i=time_inte_dict[4])

                time_inteval = 4
                pred_mean, cur_seq_std = self.exp_map(torch.cat([pred_y[:, -time_inteval*self.hid_S:-(time_inteval-1)*self.hid_S],
                                                                 pred_y_std[:, -time_inteval*self.hid_S:-(time_inteval-1)*self.hid_S]], 1),
                                                  temp_out)

            pred_y = torch.cat([pred_y, pred_mean], 1)  # **********************************
            pred_y_std = torch.cat([pred_y_std, cur_seq_std], 1)

        return pred_y[:, self.args.in_len_val*self.hid_S:], pred_y_std[:, self.args.in_len_val*self.hid_S:]#, torch.cat(pred_mean_list, 1), torch.cat(pred_log_var_list, 1)

    def forward_recur(self, x, const_emb, time_data, device, hid_i=0, **kwargs):   # const_emb coube be a const, but the time is changed
        B, C, H, W = x.shape
        time_emb = self.time_embedding(time_data.reshape(B, -1))[..., None, None].repeat(1, 1, H, W)
        Y_mean = checkpoint(self.hid[hid_i], torch.cat([x, time_emb], 1), use_reentrant=False)

        return Y_mean

    def sampling(self, z_mean, z_log_var, device, name=None):
        epsilon = torch.randn(z_mean.shape).to(device)
        # print(f'sampling {name}', torch.max(z_log_var / 2), torch.min(z_log_var / 2))
        return z_mean + torch.exp((z_log_var + self.args.z_log_var_minus) / 2) * epsilon

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 drop_rate=0.0):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation, padding_mode='circular'),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, padding_mode='circular')

        self.norm = nn.GroupNorm(2, out_channels)   # group number: 2
        # self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(p=drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            torch.nn.init.trunc_normal_(m.weight, std=.02)  # math.sqrt(2.0 / self.in_channels)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        if self.drop_rate>0:
            y = self.dropout(y)
        return y   # try more conv and resnet


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 drop_rate=0.0):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace, drop_rate=drop_rate)
        self.conv2 = BasicConv2d(C_in, C_out, kernel_size=1, stride=stride,
                                upsampling=upsampling, padding=0,
                                act_norm=act_norm, act_inplace=act_inplace, drop_rate=drop_rate)
        self.cat_conv = nn.Conv2d(C_out*2, C_out, 1)

        self.res_conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0,
                                  padding_mode='circular') if C_in != C_out else nn.Identity()

    def _forward(self, x):
        # x0 = x.clone()
        y = self.conv(x)
        y2 = self.conv2(x)
        return self.cat_conv(torch.cat([y, y2], 1)) + self.res_conv(x)

    def forward(self, x):
        # 浣跨敤checkpoint锛屼笉淇濆瓨涓棿婵€娲诲€?
        return checkpoint(self._forward, x, use_reentrant=False)


def sampling_generator(N, reverse=False):
    samplings = [False, False] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, C_in, era5_ch, C_hid, N_S, spatio_kernel, H=64, W=32, H_d=120, W_d=60,
                 act_inplace=True,
                 tar_dim=None, drop_rate=0.0, n_head=2, patch_size=4, args=None):
        samplings = sampling_generator(N_S)
        # print(samplings)
        super(Encoder, self).__init__()
        self.tar_dim = tar_dim
        self.C_hid_half = int(C_hid / 2)
        coords_dim = 2
        dem_dim = 1
        self.args = args

        tmp_ch = (C_in + C_hid) // 2 if ((C_in+C_hid)//2)%2 == 0 else (C_in+C_hid)//2 + 1
        self.enc = nn.Sequential(
              ConvSC(C_in+coords_dim+dem_dim, tmp_ch, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace, drop_rate=drop_rate),
              ConvSC(tmp_ch, C_hid, spatio_kernel, downsampling=samplings[1],
                   act_inplace=act_inplace, drop_rate=drop_rate),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, drop_rate=drop_rate) for s in samplings[2:]]
        )

        self.enc_era5 = nn.Sequential(
              ConvSC(era5_ch+coords_dim+dem_dim, 64, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace, drop_rate=drop_rate),
              ConvSC(64, C_hid, spatio_kernel, downsampling=samplings[1],
                   act_inplace=act_inplace, drop_rate=drop_rate),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, drop_rate=drop_rate) for s in samplings[2:]]
        )

        self.concat_conv = nn.Conv2d(C_hid*2, C_hid, 3, padding=1, padding_mode='circular')

        self.H, self.W, self.H_d, self.W_d = H, W, H_d, W_d
        # print('stride', int(H/H_d))
        if H != H_d and W != W_d:
            self.pre_process0 = nn.Conv2d(C_hid, C_hid, kernel_size=9, padding=4, padding_mode='circular', groups=C_hid,
                                          stride=int(H/H_d), dilation=int((H/H_d)/2))

        self.post_process_mean = nn.Conv2d(C_hid, C_hid, kernel_size=5, padding=2, padding_mode='circular')
        self.post_process_std = nn.Conv2d(C_hid, C_hid, kernel_size=5, padding=2, padding_mode='circular')


    def forward(self, x_in):  # B*4, 3, 128, 128
        # print(x.shape)
        era5_x, x, coords, dem_data = x_in

        B, C, H, W = x.shape

        latent = self.enc[0](torch.cat([x, coords[:, :, self.args.border_tar[0]:-self.args.border_tar[0],
                                           self.args.border_tar[1]:-self.args.border_tar[1]],
                                        dem_data[:, :, self.args.border_tar[0]:-self.args.border_tar[0],
                                           self.args.border_tar[1]:-self.args.border_tar[1]]], 1))
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        if self.H != self.H_d and self.W != self.W_d:
            latent = self.pre_process0(latent)
        if latent.shape[-2] != self.H_d or latent.shape[-1] != self.W_d:
            latent = F.interpolate(latent, size=[self.H_d, self.W_d], mode='bilinear')

        latent_era5 = self.enc_era5[0](torch.cat([era5_x,
                                                  F.interpolate(coords, size=[era5_x.shape[-2], era5_x.shape[-1]], mode='nearest'),
                                                  F.interpolate(dem_data, size=[era5_x.shape[-2], era5_x.shape[-1]],
                                                                mode='bilinear')], 1))
        for i in range(1, len(self.enc_era5)):
            latent_era5 = self.enc_era5[i](latent_era5)

        up_latent_era5 = F.interpolate(latent_era5, size=[self.H_d, self.W_d], mode='bilinear')
        result = self.concat_conv(torch.cat([latent, up_latent_era5], 1))

        return self.post_process_mean(result), F.softplus(self.post_process_std(result)) #F.softmax(result, dim=1)    #, enc1


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, H=64, W=32, H_d=64, W_d=32, act_inplace=True, min_rain=0.1,
                 log_min=0.1, n_head=2, patch_size=4, device=None, args=None):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.args = args

        self.H, self.W, self.H_d, self.W_d = H, W, H_d, W_d
        assert C_hid % n_head == 0
        self.dec = nn.Sequential(
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace, drop_rate=args.drop),
            SelfAttention_patch(in_channel=C_hid, n_head=n_head, norm_groups=C_hid//4, patch_size=patch_size),
            nn.Conv2d(C_hid, 16, kernel_size=3, padding=1, padding_mode='circular')
        )

        self.pre_process1 = nn.Conv2d(1, 1, kernel_size=5, padding=2, padding_mode='circular')


    def forward(self, hid_mean, hid_std, enc1=None, device=None):
        Y = self.dec(hid_mean)  #  + enc1

        if self.H != self.H_d and self.W != self.W_d:

            Y = F.pixel_shuffle(Y, 4)
            Y = checkpoint(self.pre_process1, Y, use_reentrant=False)  # Y鐨勮繛缁垎甯冭礋璐ｆā鎷熼檷闆ㄧ殑鍒嗗竷

            B, _, H, W = hid_std.shape
            x = hid_std.permute(0, 2, 3, 1).reshape(B * H, W, -1)  # [B*H, W, C]
            x = F.interpolate(x, size=16, mode='linear')  # [B*H, W, c]
            x = x.reshape(B, H, W, 16).permute(0, 3, 1, 2)  # [B, c, H, W]
            Y_std = F.pixel_shuffle(x, 4)
        else:
            Y = self.pre_process1(Y)

        return Y, Y_std

class Decoder_recon(nn.Module):
    """Decoder"""
    def __init__(self, C_hid, era5_size, fy4b_size, N_S, spatio_kernel, H=64, W=64, act_inplace=True, min_rain=0.1, device=None, drop_rate=0.0):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder_recon, self).__init__()
        self.min_rain = min_rain
        self.era5_size = era5_size
        self.fy4b_size = fy4b_size

        self.readout_era5 = nn.Sequential(
            nn.Conv2d(C_hid, (era5_size[0]+C_hid)//2, 3, padding=1, padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.Conv2d((era5_size[0]+C_hid)//2, era5_size[0], 3, padding=1, padding_mode='circular'))
        self.readout_fy4b = nn.Sequential(
            nn.Conv2d(C_hid, (fy4b_size[0]+C_hid)//2, 3, padding=1, padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.Conv2d((fy4b_size[0]+C_hid)//2, fy4b_size[0], 3, padding=1, padding_mode='circular'))

    def forward(self, Y, enc1=None):

        recon_era5 = F.interpolate(Y, size=self.era5_size[1:], mode='bilinear')
        recon_era5 = checkpoint(self.readout_era5, recon_era5, use_reentrant=False)

        recon_fy4b = F.interpolate(Y, size=self.fy4b_size[1:], mode='bilinear')
        recon_fy4b = checkpoint(self.readout_fy4b, recon_fy4b, use_reentrant=False)

        return recon_era5, recon_fy4b

