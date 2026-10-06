import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# from timm.models.layers import DropPath, trunc_normal_
import math
import numpy as np


class ConvGRU(nn.Module):
    def __init__(self, in_ch_cur, in_ch_past, in_ch_emb, out_ch, num_block_cur=2, num_block_past=4, hid_S=32, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.1, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, args=None, **kwargs):
        super(ConvGRU, self).__init__()

        self.in_ch_cur = in_ch_cur

        self.time_embedding = nn.Sequential(
            nn.Linear((args.input_time_length+args.pre_seq_length)*args.time_emb_num, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),   #  hid_S  ***
            nn.LeakyReLU(),
            nn.Linear(256, hid_S)
        )

        # self.conv_emb = Resnet(in_ch_emb, out_ch, kernel_size=3, stride=1,
        #          padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
        #          act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)

        self.conv_cur_emb = Resnet(in_ch_emb+in_ch_cur, in_ch_cur, kernel_size=3, stride=1,
                                   padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                                   act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)

        self.conv_past_emb = Resnet(in_ch_emb+in_ch_past, in_ch_past, kernel_size=3, stride=1,
                                   padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                                   act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)

        self.conv_x_z = Resnet(in_ch_cur, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)
        self.conv_h_z = Resnet(in_ch_past, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_past)

        self.conv_x_r = Resnet(in_ch_cur, in_ch_past, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)
        self.conv_h_r = Resnet(in_ch_past, in_ch_past, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_past)

        self.conv_h_t_1 = Resnet(in_ch_past, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_past)

        self.conv = Resnet(in_ch_cur, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_cur)
        self.conv_u = Resnet(in_ch_past, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_past)

        self.conv_out = Resnet(out_ch, out_ch, kernel_size=3, stride=1,
                 padding=0, drop_rate=args.drop, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=act_inplace, num_block=num_block_past)

    def forward(self, x0, const_emb, time_data, **kwargs):
        # x represents the corrent var, h_t_1 represents the past vars
        B, _, H, W = x0.size()
        x = x0[:, -self.in_ch_cur:, ...].clone()
        h_t_1 = x0[:, :-self.in_ch_cur, ...].clone()

        time_emb = self.time_embedding(time_data.reshape(B, -1))[..., None, None].repeat(1, 1, H, W)
        emb = torch.cat([const_emb, time_emb], 1)
        # emb = self.conv_emb(torch.cat([const_emb, time_emb], 1))
        # print(x.shape)
        # print(emb.shape)
        x = self.conv_cur_emb(torch.cat([x, emb], 1))
        h_t_1 = self.conv_past_emb(torch.cat([h_t_1, emb], 1))

        # 当前的权重
        z_t = F.sigmoid(self.conv_x_z(x) + self.conv_h_z(h_t_1))

        # 过去的权重
        r_t = F.sigmoid((self.conv_x_r(x) + self.conv_h_r(h_t_1)))

        # 更新当前状态
        h_hat_t = F.tanh(self.conv(x) + self.conv_u(torch.mul(r_t, h_t_1)))   # mul操作对象的维度

        h_t = torch.mul((1 - z_t), self.conv_h_t_1(h_t_1)) + torch.mul(z_t, h_hat_t)   # mul操作对象的维度

        y = self.conv_out(h_t)

        return y#, h_t


class Discriminator(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Discriminator, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size, stride=1, padding=1, padding_mode='circular')
        # self.conv1.bias.data = self.conv1.bias.data.to(torch.float16)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(64, 128, kernel_size, stride=2, padding=1, padding_mode='circular')
        # self.conv2.bias.data = self.conv2.bias.data.to(torch.float16)
        # 第三个卷积层
        self.conv3 = nn.Conv2d(128, 256, kernel_size, stride=2, padding=1, padding_mode='circular')
        # self.conv3.bias.data = self.conv3.bias.data.to(torch.float16)
        self.conv4 = nn.Conv2d(256, 512, kernel_size, stride=2, padding=1, padding_mode='circular')
        # self.conv4.bias.data = self.conv4.bias.data.to(torch.float16)
        # 最终卷积层输出指定数量的输出波段
        self.final_conv = nn.Conv2d(512, output_channels, kernel_size, stride=1, padding=1, padding_mode='circular')
        # self.final_conv.bias.data = self.final_conv.bias.data.to(torch.float16)
        # 初始化函数定义为self的方法
        # self.initialize_bias_to_float16()

    def initialize_bias_to_float16(self):
        # 遍历模型的每个参数
        for name, param in self.named_parameters():
            # 判断参数类型为偏置参数且数据类型为torch.float32
            if 'bias' in name: # and param.data.dtype == torch.float32:
                # 将偏置初始化为float16类型
                param.data = param.data.to(torch.float16)

    def forward(self, x):
        # 输入数据经过卷积层和激活函数
        # print(x.dtype)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 最终卷积层输出
        x = F.sigmoid(self.final_conv(x))

        return x

class Res_net_conv(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=3):
        super(Res_net_conv, self).__init__()

        padding_mode = 'circular'
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.SiLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1,
                               padding_mode=padding_mode,
                               padding=int(kernelsize // 2))

    def forward(self, x):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2 + x

class AdversarialLoss(nn.Module):
    def __init__(self, discriminator_network, loss_type='binary_cross_entropy'):
        super(AdversarialLoss, self).__init__()
        self.discriminator_network = discriminator_network

        # Define the loss function
        if loss_type == 'binary_cross_entropy':
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss_type == 'mean_squared_error':
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type. Supported types: 'binary_cross_entropy', 'mean_squared_error'")

        # Define optimizer for the discriminator
        self.discriminator_optimizer = optim.Adam(self.discriminator_network.parameters(), lr=0.001)

    def forward(self, real_values, predicted_values):
        # Forward pass through the discriminator network
        real_outputs = self.discriminator_network(real_values)
        predicted_outputs = self.discriminator_network(predicted_values)

        # Compute the discriminator loss
        real_labels = torch.ones_like(real_outputs)
        predicted_labels = torch.zeros_like(predicted_outputs)

        discriminator_loss = self.loss_function(real_outputs, real_labels) + self.loss_function(predicted_outputs, predicted_labels)

        # Update the discriminator parameters
        discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()

        # Compute the generator loss
        generator_labels = torch.ones_like(predicted_outputs)  # Generator wants the discriminator to output 1 for generated samples
        generator_loss = self.loss_function(predicted_outputs, generator_labels)

        return generator_loss

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)  # each group will have 4 channels
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

import torch.nn.functional as F

class SelfAttention_patch(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, patch_size=4):
        super().__init__()
        self.n_head = n_head
        self.patch_size = patch_size
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        ps = self.patch_size

        norm = self.norm(input)
        qkv = self.qkv(norm)
        qkv = qkv.unfold(2, ps, ps).unfold(3, ps, ps)\
                 .permute(0, 1, 4, 5, 2, 3)\
                 .reshape(batch, n_head, head_dim * 3, ps * ps, -1)\
                 .permute(0, 1, 3, 2, 4)                 # b n p c h
        q, k, v = qkv.chunk(3, dim=3)

        # (b n p c h) -> (b n p h c): patch序号h当序列维, 通道c当特征维
        q = q.permute(0, 1, 2, 4, 3)
        k = k.permute(0, 1, 2, 4, 3)
        v = v.permute(0, 1, 2, 4, 3)
        out = F.scaled_dot_product_attention(q, k, v)    # 关键: 保持旧温度, 与epoch90权重兼容, 重新训练时可删除: , scale=1.0 / math.sqrt(channel)
        out = out.permute(0, 1, 2, 4, 3)        # 回到 b n p c h

        out = out.permute(0, 1, 3, 2, 4)\
                 .reshape(batch, channel, ps, ps, height // ps, width // ps)\
                 .permute(0, 1, 2, 4, 3, 5)\
                 .reshape(batch, channel, height, width)
        out = self.out(out)
        return out + input


class Resnet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 drop_rate=0.1,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 num_block=12,
                 norm_band_num=4,
                 patch_size=4,
                 step=2):
        super(Resnet, self).__init__()
        self.num_block = num_block

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, (in_channels+out_channels) // 2, 3, padding=1, padding_mode='circular'),
            nn.LeakyReLU(inplace=act_inplace),
            nn.Conv2d((in_channels + out_channels) // 2, out_channels, 3, padding=1, padding_mode='circular')
        )

        scale_f = [1]*7 + [2]*5
        ope = ['down']*2 + ['no']*8 + ['up']*2
        self.module = nn.ModuleList([Resnet_block(out_channels*scale_f[i], out_channels, kernel_size=kernel_size, stride=stride,
                 padding=padding, drop_rate=drop_rate, act_inplace=act_inplace,
                                                  norm_band_num=norm_band_num, patch_size=patch_size, step=step, ope=ope[i]) for i in range(num_block)])

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x0 = x.clone()
        # print(x.size())
        x = self.enc(x)
        skip_connections = []
        skip_connections.append(x)

        for i in range(self.num_block // 2):
            # print(i)
            x = self.module[i](x)
            skip_connections.append(x)

        skip_connections.pop()
        x = self.module[self.num_block // 2](x)
        for i in range(self.num_block // 2 + 1, self.num_block):
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = self.module[i](x)

        return x + self.res_conv(x0)


class Resnet_block(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3, stride=1, padding=0, drop_rate=0.1,
                 dilation=1, upsampling=False, act_norm=False, act_inplace=True,
                 norm_band_num=4, patch_size=4, step=2, ope=None):
        super(Resnet_block, self).__init__()
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2),
                      padding_mode='circular'),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=act_inplace),
            nn.Dropout(p=drop_rate)]
        )
        self.res_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                  padding_mode='circular')])
        if ope == 'down':
            self.conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=int(kernel_size/2),
                      padding_mode='circular'))

            self.res_conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=int(kernel_size/2),
                      padding_mode='circular'))
        elif ope == 'up':
            self.conv.append(
                nn.Upsample(scale_factor=2, mode='bilinear'))
            self.conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int(kernel_size/2),
                      padding_mode='circular'))

            self.res_conv.append(
                nn.Upsample(scale_factor=2, mode='bilinear'))
            self.res_conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int(kernel_size/2),
                      padding_mode='circular'))
        self.conv = nn.Sequential(*self.conv)
        # self.pixel_att = SelfAttention(out_channels, n_head=3, norm_groups=out_channels//norm_band_num)
        self.patch_att = SelfAttention_patch(out_channels, n_head=3,
                                             norm_groups=out_channels//norm_band_num, patch_size=patch_size)
        self.win_att = SelfAttention_win(out_channels, n_head=3,
                                             norm_groups=out_channels//norm_band_num, patch_size=patch_size, step=step)

        # self.concat = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.concat = CrossAttention(out_channels)

        self.res_conv = nn.Sequential(*self.res_conv)
        # self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular') if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x0 = self.conv(x)
        x_p = self.patch_att(x0)
        x_w = self.win_att(x0)
        return self.concat(x_p, x_w) + self.res_conv(x)  # 调换二者顺序
