import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchinfo
from einops import rearrange
from .pwac import filter_indice, dispatch_indice, permute, inverse_permute, batched_matmul_conv

def convolution_by_cluster(patches: torch.Tensor, indice: torch.Tensor, weight: torch.Tensor, bias=None):
    """
    Args:
        patches: (batch_size, patch_num, patch_dims)
        indice: (batch_size, patch_num)
        weight: (batch_size, cluster_num, in_channels * kernel_area, out_channels)
        bias: (batch_size, cluster_num, out_channels)
    Returns:
        res: (batch_size, patch_num, out_channels)
    """
    b = patches.shape[0]
    k = weight.shape[1]

    patches = rearrange(patches, "b s f -> (b s) f")
    weight = rearrange(weight, "b k f cout -> (b k) f cout")
    indice = indice + torch.arange(b, device=indice.device).view(-1, 1) * k
    indice = rearrange(indice, "b hw -> (b hw)")
    if bias is not None:
        bias = rearrange(bias, "b k cout -> (b k) cout")

    indice_perm, padded_patch_num, cluster_size_sorted, permuted_offset, cluster_perm, batch_height = dispatch_indice(
        indice, b * k)
    input_permuted = permute(patches, indice_perm, padded_patch_num)
    output_permuted = batched_matmul_conv(
        input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias)
    output = inverse_permute(output_permuted, indice_perm)

    return rearrange(output, "(b hw) cout -> b hw cout", b=b)

class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                          groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels*2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=2, padding=1),
                # nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels*2, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)

class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels//2, in_channels//2, 3, 1,
                          1, groups=in_channels//2, bias=False),
                nn.Conv2d(in_channels//2, in_channels//2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1)

    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x

class mask_generator(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(mask_generator, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=1, groups=1, bias=True)

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        mask_soft = torch.sigmoid(x)   #[b, c, h, w]
        mask_soft_flat = torch.mean(mask_soft, dim=1, keepdim=True)   #[b, 1, h, w]
        mean = mask_soft_flat.mean(dim=(2, 3), keepdim=True)
        std = mask_soft_flat.std(dim=(2, 3), keepdim=True)
        alpha = 2.0
        threshold = mean + alpha * std
        mask_hard = (mask_soft_flat > threshold).float()
        #ratio = mask_hard.mean(dim=(2, 3)).view(b)
        #print("1的占比：", ratio.tolist())
        return mask_soft, mask_hard   #[b, 1, h, w]

class atw_generator(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels):
        super(atw_generator, self).__init__()
        self.in_dim = in_channels
        self.hidden_dim = in_channels // 2
        self.kernel_size = kernel_size
        self.area_dim = kernel_size * kernel_size
        self.out_dim = out_channels
        self.fc1 = nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.fc2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.fc_cin = nn.Conv2d(self.hidden_dim, self.in_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.fc_cout = nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.fc_area = nn.Conv2d(self.hidden_dim, self.area_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x, mask):   # x:b, h*w, cin*k*k   mask:b, 1, h, w
        b, _, h, w = mask.shape
        _, hw, _ = x.shape
        indice = mask.squeeze(1).view(b, -1).long()   # [B, H*W]
        x0 = F.fold(
            x.permute(0, 2, 1),
            output_size=(h, w),
            kernel_size=self.kernel_size,
            padding=1,
            stride=1
        )  # [b, c, h, w]
        x_flat = x0.flatten(2).permute(0, 2, 1)  # [b, h*w, cin]
        weight1 = self.fc1.weight.view(self.hidden_dim, -1).permute(1, 0)   #[cout, cin, k, k]->[cout, cin*k*k]->[cin*k*k, cout]
        weight1 = weight1.unsqueeze(0).expand(b, -1, -1).unsqueeze(1)   #[cin*k*k, cout]->[1, cin*k*k, cout]->[b, cin*k*k, cout]->[b, 1, cin*k*k, cout]
        weight1_zero = torch.zeros_like(weight1, device = x.device, dtype=x.dtype)
        weight1 = torch.cat([weight1_zero, weight1], dim=1)
        bias1 = self.fc1.bias.unsqueeze(0).expand(b, -1).unsqueeze(1)  # [B,1,Cout]
        bias1_zero = torch.zeros_like(bias1, device=x.device, dtype=x.dtype)
        bias1 = torch.cat([bias1_zero, bias1], dim=1)
        atw = convolution_by_cluster(x_flat, indice, weight1, bias1)   #[b, h*w, hidden_dim]
        atw = F.relu(atw)

        weight2 = self.fc2.weight.view(self.hidden_dim, -1).permute(1, 0)
        weight2 = weight2.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)
        weight2_zero = torch.zeros_like(weight2, device = x.device, dtype=x.dtype)
        weight2 = torch.cat([weight2_zero, weight2], dim=1)
        bias2 = self.fc2.bias.unsqueeze(0).expand(b, -1).unsqueeze(1)
        bias2_zero = torch.zeros_like(bias2, device=x.device, dtype=x.dtype)
        bias2 = torch.cat([bias2_zero, bias2], dim=1)
        atw = convolution_by_cluster(atw, indice, weight2, bias2)   #[b, h*w, hidden_dim]
        atw = F.relu(atw)

        weight_cin = self.fc_cin.weight.view(self.in_dim, -1).permute(1, 0)
        weight_cin = weight_cin.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)
        weight_cin_zero = torch.zeros_like(weight_cin, device = x.device, dtype=x.dtype)
        weight_cin = torch.cat([weight_cin_zero, weight_cin], dim=1)
        bias_cin = self.fc_cin.bias.unsqueeze(0).expand(b, -1).unsqueeze(1)
        bias_cin_zero = torch.zeros_like(bias_cin, device=x.device, dtype=x.dtype)
        bias_cin = torch.cat([bias_cin_zero, bias_cin], dim=1)
        atw_cin = convolution_by_cluster(atw, indice, weight_cin, bias_cin)   #[b, h*w, in_dim]
        atw_cin = F.sigmoid(atw_cin)

        weight_cout = self.fc_cout.weight.view(self.out_dim, -1).permute(1, 0)
        weight_cout = weight_cout.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)
        weight_cout_zero = torch.zeros_like(weight_cout, device=x.device, dtype=x.dtype)
        weight_cout = torch.cat([weight_cout_zero, weight_cout], dim=1)
        bias_cout = self.fc_cout.bias.unsqueeze(0).expand(b, -1).unsqueeze(1)
        bias_cout_zero = torch.zeros_like(bias_cout, device=x.device, dtype=x.dtype)
        bias_cout = torch.cat([bias_cout_zero, bias_cout], dim=1)
        atw_cout = convolution_by_cluster(atw, indice, weight_cout, bias_cout)  # [b, h*w, out_dim]
        atw_cout = F.sigmoid(atw_cout)

        weight_area = self.fc_area.weight.view(self.area_dim, -1).permute(1, 0)
        weight_area = weight_area.unsqueeze(0).unsqueeze(1).expand(b, -1, -1, -1)
        weight_area_zero = torch.zeros_like(weight_area, device=x.device, dtype=x.dtype)
        weight_area = torch.cat([weight_area_zero, weight_area], dim=1)
        bias_area = self.fc_area.bias.unsqueeze(0).expand(b, -1).unsqueeze(1)
        bias_area_zero = torch.zeros_like(bias_area, device=x.device, dtype=x.dtype)
        bias_area = torch.cat([bias_area_zero, bias_area], dim=1)
        atw_area = convolution_by_cluster(atw, indice, weight_area, bias_area)  # [b, h*w, k*k]
        atw_area = F.sigmoid(atw_area)

        atw_spatial = atw_cin.unsqueeze(3) * atw_area.unsqueeze(2)   #[b, h*w, cin, k*k]
        atw_spatial = atw_spatial.view(b, hw, -1)

        return atw_spatial, atw_cout   #[b, h*w, cin*k*k], [b, h*w, cout]

class TuckerConv(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels):
        super(TuckerConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.inner_channels = in_channels // 2
        self.r1 = 3
        self.r2 = 3
        self.r3 = 2
        self.head_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.inner_channels),
            nn.LeakyReLU()
        )
        self.U_net = nn.Sequential(
            nn.Linear(self.inner_channels, self.in_channels * self.r1),
            nn.Sigmoid()
        )
        self.V_net = nn.Sequential(
            nn.Linear(self.inner_channels, self.out_channels * self.r2),
            nn.Sigmoid()
        )
        self.W_net = nn.Sequential(
            nn.Linear(self.inner_channels, self.kernel_size*self.kernel_size*self.r3),
            nn.Sigmoid()
        )
        self.core = nn.Parameter(torch.randn(self.r1, self.r2, self.r3)*0.01, requires_grad=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=1)

    def forward(self, x):
        b = x.shape[0]
        x_pooled = self.gap(x)   #[b, c, 1, 1]
        x_pooled = x_pooled.squeeze(-1).squeeze(-1)   #[b, c]
        x_pooled = self.head_mlp(x_pooled)   #[b, inner_channels]
        U = self.U_net(x_pooled)   #[b, cin*r1]
        V = self.V_net(x_pooled)   #[b, cout*r2]
        W = self.W_net(x_pooled)   #[b, k*k*r3]
        U = U.view(b, self.in_channels, self.r1).mean(0)   #[cin, r1]
        V = V.view(b, self.out_channels, self.r2).mean(0)   #[cout, r2]
        W = W.view(b, self.kernel_size*self.kernel_size, self.r3).mean(0)  #[k*k, r3]
        core = torch.einsum('abc, ia, ob, sc->ois', self.core, U, V, W)   #[cout, cin, k*k]
        core = core.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        core = core.to(x.dtype).to(x.device)   #[cout, cin, k, k]
        weight = self.conv.weight * (core + 1)
        bias = self.conv.bias
        output = F.conv2d(x, weight, bias, stride=1, padding=1, groups=1)
        return output

class AT(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels):
        super(AT, self).__init__()
        self.conv1 = TuckerConv(in_channels, kernel_size, out_channels)
        self.conv2 = TuckerConv(in_channels, kernel_size, out_channels)
        self.conv3 = TuckerConv(in_channels, kernel_size, out_channels)

    def forward(self, x):
        temp = F.leaky_relu(self.conv1(x))
        temp = F.dropout2d(temp, p=0.3)
        temp = F.leaky_relu(self.conv2(temp))
        temp = F.dropout2d(temp, p=0.3)
        output = self.conv3(temp)
        return output

class MAConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, use_bias=True):
        super(MAConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        self.mask_generator = mask_generator(self.in_channels, self.kernel_size)
        self.atw_generator = atw_generator(self.in_channels, self.kernel_size, self.out_channels)
        #掩码值为1所用的conv
        self.point_1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride,
                                      padding=0, groups=self.groups, bias=True)
        self.point_2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride,
                                      padding=0, groups=self.groups, bias=True)
        self.depth_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size,
                                      stride=self.stride, padding=self.padding, groups=self.out_channels, bias=True)
        self.depth_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size,
                                      stride=self.stride, padding=self.padding, groups=self.out_channels, bias=True)
        if self.bias == True:
            #self.bias_generator = AT(self.in_channels, self.kernel_size, self.out_channels)
            self.bias_generator = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
            )
        #掩码值为0所用的spanconv
        self.point_conv_1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride,
                                       padding=0, groups=self.groups, bias=True)
        self.point_conv_2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride,
                                       padding=0, groups=self.groups, bias=True)
        self.depth_conv_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding, groups=self.out_channels, bias=True)
        self.depth_conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding, groups=self.out_channels, bias=True)
        self.atw_cin = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels, bias=True),
            nn.Sigmoid()
        )
        self.atw_cout = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels, bias=True),
            nn.Sigmoid()
        )
        self.atw_area = nn.Sequential(
            nn.Linear(self.in_channels, self.kernel_size * self.kernel_size, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        (b, c, h, w) = x.shape
        if self.bias == True:
            bias = self.bias_generator(x)   #bias.shape = [b, cout, h, w]
#-----generate mask-----#
        mask_soft, mask_hard = self.mask_generator(x)   #mask.shape = [b, 1, h, w]
#-----soft mask as atw on all pixels-----#
        x = x * mask_soft
#-----generate the attention weight-----#
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)   #[b, cin*k*k, h*w]
        patches = x_unfold.permute(0, 2, 1)   #[b, h*w, cin*k*k]
        atw_spatial, atw_channel = self.atw_generator(patches, mask_hard)  #[b, h*w, cin*k*k], [b, h*w, cout]
        atw0 = (1 - mask_hard).view(b, 1, -1).permute(0, 2, 1).expand(-1, -1, self.in_channels * self.kernel_size * self.kernel_size)
        atw_spatial = atw_spatial + atw0
        patches_atw = patches * atw_spatial
# -----generate the low rank attention kernel-----#
        spanarea_sum = (x * (1 - mask_hard)).sum(dim=(2, 3), keepdim=True)  # [b, cin, 1, 1]
        mask_sum = (1 - mask_hard).sum(dim=(2, 3), keepdim=True)  # [b, 1 , 1, 1]
        eps = 1e-6
        span_pooled = (spanarea_sum / (mask_sum + eps)).view(b, self.in_channels)
        atw_cin = self.atw_cin(span_pooled).unsqueeze(-1).unsqueeze(-1)  # [b, cin]->[b, cin]->[b, cin, 1, 1]
        atw_cout = self.atw_cout(span_pooled).unsqueeze(1).unsqueeze(1)  # [b, cin]->[b, cout]->[b, 1, cout]->[b, 1, 1, cout]
        atw_area = self.atw_area(span_pooled).unsqueeze(1).unsqueeze(-1)  # [b, cin]->[b, k*k]->[b, 1, k*k]->[b, 1, k*k, 1]
        atw_lowrank = (atw_cin * atw_area * atw_cout).view(b, self.in_channels * self.kernel_size * self.kernel_size,
                                                           self.out_channels).unsqueeze(1)   #[b, 1, cin*k*k, cout]
#-----use pwac to deal with two kinds of pixels-----#
        LAConv_w1 = self.depth_1.weight * self.point_1.weight  # [cout, 1, k, k] * [cout, cin, 1, 1]=[cout, cin, k, k]
        LAConv_w2 = self.depth_2.weight * self.point_2.weight
        LAConv_w = (LAConv_w1 + LAConv_w2).view(self.out_channels, -1).permute(1, 0).unsqueeze(0).expand(b, -1, -1).unsqueeze(1)
        bias_la1 = self.point_1.bias + \
                     (self.point_1.weight.squeeze(-1).squeeze(-1) *
                      self.depth_1.bias.unsqueeze(1)).sum(dim=1)

        bias_la2 = self.point_2.bias + \
                     (self.point_2.weight.squeeze(-1).squeeze(-1) *
                      self.depth_2.bias.unsqueeze(1)).sum(dim=1)
        LAConv_bias = (bias_la1 + bias_la2).unsqueeze(0).expand(b, -1).unsqueeze(1)


        SpanConv_w1 = self.depth_conv_1.weight * self.point_conv_1.weight   #[cout, 1, k, k] * [cout, cin, 1, 1]=[cout, cin, k, k]
        SpanConv_w2 = self.depth_conv_2.weight * self.point_conv_2.weight
        SpanConv_w = (SpanConv_w1 + SpanConv_w2).view(self.out_channels, -1).permute(1, 0).unsqueeze(0).expand(b, -1, -1).unsqueeze(1) #[b, 1, cin*k*k, cout]
        SpanConv_w = SpanConv_w * atw_lowrank
        bias_span1 = self.point_conv_1.bias + \
                     (self.point_conv_1.weight.squeeze(-1).squeeze(-1) *
                      self.depth_conv_1.bias.unsqueeze(1)).sum(dim=1)

        bias_span2 = self.point_conv_2.bias + \
                     (self.point_conv_2.weight.squeeze(-1).squeeze(-1) *
                      self.depth_conv_2.bias.unsqueeze(1)).sum(dim=1)
        SpanConv_bias = (bias_span1 + bias_span2).unsqueeze(0).expand(b, -1).unsqueeze(1)

        kernel_by_mask = torch.cat([SpanConv_w, LAConv_w], dim=1)   #[b, 2, cin*k*k, cout]
        bias_by_mask = torch.cat([SpanConv_bias, LAConv_bias], dim=1)   #[b, 2, cout]
        indice = mask_hard.squeeze(1).view(b, h*w).long()
        #[b, 1, h, w]->[b, h, w]->[b, h*w]
        output = convolution_by_cluster(
            patches_atw, indice, kernel_by_mask, bias_by_mask
        )   #[b, h*w, cout]
        output = output.permute(0, 2, 1).view(b, self.out_channels, h, w)   #->[b, cout, h, w]

        atw_channel = atw_channel.permute(0, 2, 1).view(b, self.out_channels, h, w)  # [b, h*w, cout]->[b, cout, h*w]->[b, cout, h, w]
        atw0 = (1 - mask_hard).expand(-1, self.out_channels, -1, -1)  # [b, 1, h, w]->reverse->[b, cout, h, w]
        atw_channel = atw_channel + atw0
        output = output * atw_channel

        if self.bias == True:
            output = output + bias

        return output

#MAConv_ResBlocks
class MACRB(nn.Module):
    def __init__(self, in_channels):
        super(MACRB, self).__init__()
        self.conv1 = MAConv2D(in_channels, in_channels, 3, 1, 1, use_bias=True)
        self.conv2 = MAConv2D(in_channels, in_channels, 3, 1, 1, use_bias=True)

    def forward(self, x):        
        res = self.conv1(x)
        res = F.leaky_relu(res)
        res = self.conv2(res)
        x = x + res
        return x

#NetWork
class MANet(nn.Module):
    def __init__(self):
        super(MANet, self).__init__()
        self.head_conv = nn.Conv2d(9, 32, 3, 1, 1)
        self.RB1 = MACRB(32)
        self.down1 = ConvDown(32)
        self.RB2 = MACRB(32*2)
        self.down2 = ConvDown(32*2)
        self.RB3 = MACRB(32*4)
        self.up1 = ConvUp(32*4)
        self.RB4 = MACRB(32*2)
        self.up2 = ConvUp(32*2)
        self.RB5 = MACRB(32)
        self.tail_conv = nn.Conv2d(32, 8, 3, 1, 1)

    def forward(self, pan, lms):
        x = torch.cat([pan, lms], 1)
        x = self.head_conv(x)
        x1 = self.RB1(x)
        x2 = self.down1(x1)
        x2 = self.RB2(x2)
        x3 = self.down2(x2)
        x3 = self.RB3(x3)
        x4 = self.up1(x3, x2)
        del x2
        x4 = self.RB4(x4)
        x5 = self.up2(x4, x1)
        del x1
        x5 = self.RB5(x5)
        x5 = self.tail_conv(x5)
        sr = lms + x5
        return sr#,

if __name__ == '__main__':
    from torchinfo import summary
    N = MANet()
    pan = torch.randn(1, 1, 64, 64)
    lms = torch.randn(1, 4, 64, 64)
    summary(N, input_data = (pan, lms), device = 'cuda:0')
'''
if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    device = torch.device("cuda:0")
    model = MANet().to(device)
    model.eval()


    # 多输入模型时必须定义 input_constructor
    def input_constructor(input_res):
        pan_shape, lms_shape = input_res
        pan = torch.randn(1, *pan_shape).to(device)
        lms = torch.randn(1, *lms_shape).to(device)
        return dict(pan=pan, lms=lms)


    macs, params = get_model_complexity_info(
        model,
        input_res=((1, 64, 64), (8, 64, 64)),  # 注意：这是两个 tuple
        input_constructor=input_constructor,  # 关键：告诉 ptflops 如何构造多输入
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print(f"Total MACs: {macs}")
    print(f"Total Params: {params}")
'''
