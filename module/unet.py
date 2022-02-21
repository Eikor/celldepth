import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    else:
        padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    

    def __init__(self, in_channels, out_channels, mid_channels=None, smooth=False):
        super().__init__()
        if not mid_channels:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.smooth = smooth
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if not self.smooth:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))

        else:
            out = self.conv1(x)
            out = F.relu(self.bn1(self.kernel1(out)))         

            out = self.conv2(out)
            out = F.relu(self.bn2(self.kernel2(out)))
        return out

    def update_kernels(self, kernel_size, std):
        self.kernel1 = get_gaussian_filter(
                kernel_size=kernel_size,
                sigma=std,
                channels=self.mid_channels,
        ).cuda()
        self.kernel2 = get_gaussian_filter(
                kernel_size=kernel_size,
                sigma=std,
                channels=self.out_channels,
        ).cuda()

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, coordconv=False, smooth=False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, smooth=smooth)

    def forward(self, x):
        return self.conv(self.maxpool(x))
    
    def update_kernels(self, kernel_size, std):
        self.conv.update_kernels(kernel_size, std)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, coordconv=False, smooth=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, smooth=smooth)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, smooth=smooth)



    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    def update_kernels(self, kernel_size, std):
        self.conv.update_kernels(kernel_size, std)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1):
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, coordconv=False, smooth=False):
        super(UNet, self).__init__()
        self.smooth = smooth
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_offset = False
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64, coordconv, smooth=smooth)
        self.down2 = Down(64, 128, coordconv, smooth=smooth)
        self.down3 = Down(128, 256, coordconv, smooth=smooth)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor, smooth=smooth)
        self.up1 = Up(512, 256 // factor, bilinear, smooth=smooth)
        self.up2 = Up(256, 128 // factor, bilinear, smooth=smooth)
        self.up3 = Up(128, 64 // factor, bilinear, smooth=smooth)
        self.up4 = Up(64, 32, bilinear, smooth=smooth)
        self.outc = OutConv(32, n_classes)

    
    def forward(self, x):
        if self.smooth:
            x1 = self.kernel1(self.inc(x))
        else:
            x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)      
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def update_kernels(self, kernel_size, std):
        self.kernel1 = get_gaussian_filter(
                kernel_size=kernel_size,
                sigma=std,
                channels=32
        ).cuda()
        self.down1.update_kernels(kernel_size, std)
        self.down2.update_kernels(kernel_size, std)
        self.down3.update_kernels(kernel_size, std)
        self.down4.update_kernels(kernel_size, std)
        self.up1.update_kernels(kernel_size, std)
        self.up2.update_kernels(kernel_size, std)
        self.up3.update_kernels(kernel_size, std)
        self.up4.update_kernels(kernel_size, std)

class Aux_net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, aux_channel=8):
        super(Aux_net, self).__init__()
        # self.smooth = smooth
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        aux_channels = 2**torch.arange(5)*aux_channel
        # self.deep_offset = False
        self.inc = DoubleConv(n_channels, aux_channel)
        self.down1 = Down(aux_channels[0], aux_channels[1], False, smooth=False)
        self.down2 = Down(aux_channels[1], aux_channels[2], False, smooth=False)
        self.down3 = Down(aux_channels[2], aux_channels[3], False, smooth=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(aux_channels[3], aux_channels[4] // factor, smooth=False)
        self.up1 = Up(aux_channels[4], aux_channels[3] // factor, bilinear, smooth=False)
        self.up2 = Up(aux_channels[3], aux_channels[2] // factor, bilinear, smooth=False)
        self.up3 = Up(aux_channels[2], aux_channels[1] // factor, bilinear, smooth=False)
        self.up4 = Up(aux_channels[1], aux_channels[0], bilinear, smooth=False)
        self.outc = OutConv(aux_channels[0], n_classes)

    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits