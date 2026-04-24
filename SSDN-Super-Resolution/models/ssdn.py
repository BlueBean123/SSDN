import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.fft


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class SobelOperator(nn.Module):
    def __init__(self, in_channels):
        super(SobelOperator, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('weight_x', sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('weight_y', sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.groups = in_channels

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.groups)
        grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.groups)
        return torch.abs(grad_x) + torch.abs(grad_y)


class CASFEM(nn.Module):
    def __init__(self, nf=64, in_nc=3, reduction=16):
        super(CASFEM, self).__init__()
        self.sobel = SobelOperator(in_nc)
        mip = max(8, nf // reduction)
        self.conv_grad_embed = nn.Sequential(
            nn.Conv2d(in_nc, mip, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mip),
            h_swish()
        )
        self.conv_1x1 = nn.Conv2d(nf, mip, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, nf, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, nf, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(nf, nf, 3, 1, 1)
        initialize_weights([self.conv_grad_embed, self.conv_1x1, self.conv_h, self.conv_w, self.conv_out], 0.1)

    def forward(self, x_feat, x_raw):
        identity = x_feat
        n, c, h, w = x_feat.size()

        grad_raw = self.sobel(x_raw)
        grad_embed = self.conv_grad_embed(grad_raw)

        x_h = F.adaptive_avg_pool2d(x_feat, (h, 1))
        x_w = F.adaptive_avg_pool2d(x_feat, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn(self.conv_1x1(y)))

        g_h = F.adaptive_avg_pool2d(grad_embed, (h, 1))
        g_w = F.adaptive_avg_pool2d(grad_embed, (1, w)).permute(0, 1, 3, 2)
        g_y = torch.cat([g_h, g_w], dim=2)

        y_fused = y + g_y

        x_h, x_w = torch.split(y_fused, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = x_feat * a_h * a_w
        return self.conv_out(out) + identity


class MultiScaleSpectralBiasGenerator(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(MultiScaleSpectralBiasGenerator, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.fusion = nn.Linear(in_channels, output_dim)
        initialize_weights([self.conv1x1, self.conv3x3, self.fusion], 0.1)

    def forward(self, x_amp):
        f1 = self.act(self.conv1x1(x_amp)).mean(dim=(2, 3))
        f2 = self.act(self.conv3x3(x_amp)).mean(dim=(2, 3))
        f_cat = torch.cat([f1, f2], dim=1)
        bias = self.fusion(f_cat)
        return bias


class WindowAttentionWithSpectralBias(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.bias_generator = MultiScaleSpectralBiasGenerator(
            in_channels=dim,
            output_dim=num_heads * (window_size ** 4)
        )
        initialize_weights([self.qkv, self.proj], 0.1)

    def forward(self, x):
        B_, N, C = x.shape
        H = W = self.window_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        x_img = x.view(B_, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_fft = torch.fft.rfft2(x_img.float(), norm='backward')
        x_amp = torch.abs(x_fft)
        spec_bias = self.bias_generator(x_amp).view(B_, self.num_heads, N, N)

        attn = attn + spec_bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        return x


class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(ConvFFN, self).__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        initialize_weights([self.fc1, self.dwconv, self.fc2], 0.1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SpectralGatingBranch(nn.Module):
    def __init__(self, nf=64):
        super(SpectralGatingBranch, self).__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(nf, nf // 4, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf // 4, nf, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        initialize_weights([self.freq_gate], 0.1)

    def forward(self, x):
        x_in = x.float()
        x_fft = torch.fft.rfft2(x_in, norm='backward')
        amp = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        gate = self.freq_gate(amp)
        amp = amp * gate

        fft_modulated = amp * torch.exp(1j * pha)
        global_feat = torch.fft.irfft2(fft_modulated, s=x.shape[-2:], norm='backward')
        return global_feat.to(dtype=x.dtype)


class DFIB(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super(DFIB, self).__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionWithSpectralBias(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.conv_ffn = ConvFFN(dim, dim * 2)

        self.spectral_gating = SpectralGatingBranch(dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H // self.window_size, self.window_size, W // self.window_size,
                                       self.window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape

        spectral_feat = self.spectral_gating(x)

        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        Hp, Wp = x_padded.shape[2], x_padded.shape[3]

        x_windows = self.window_partition(x_padded)
        x_windows = x_windows + self.attn(self.norm1(x_windows))

        x_trans = self.window_reverse(x_windows, Hp, Wp)
        x_trans = x_trans[:, :, :H, :W]

        x_trans_perm = x_trans.permute(0, 2, 3, 1)
        x_trans_perm = x_trans_perm + self.conv_ffn(self.norm2(x_trans_perm))
        x_trans_out = x_trans_perm.permute(0, 3, 1, 2)

        out = x_trans_out + self.alpha * spectral_feat + shortcut
        return out

class SSDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(SSDN, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.casfem = CASFEM(nf=nf, in_nc=in_nc, reduction=16)

        trunk = []
        for i in range(nb):
            if i > 0 and (i + 1) % 4 == 0:
                trunk.append(DFIB(dim=nf, window_size=8))
            else:
                trunk.append(ResidualBlock_noBN(nf=nf))
        self.recon_trunk = nn.Sequential(*trunk)

        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat_initial = self.conv_first(x)
        feat_shallow = self.casfem(feat_initial, x)
        deep_out = self.recon_trunk(feat_shallow)
        out = deep_out + feat_shallow

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        out += base
        return out