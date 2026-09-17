import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import hyp2f1
from functools import partial


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        use_conv,
        out_channels=None,
        padding=0,
        padding_mode="zeros",
    ):
        """_summary_

        Args:
            in_channels (int): _description_
            use_conv (bool): _description_
            out_channels (Optional[int], optional): _description_. Defaults to None.
            padding (int, optional): _description_. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        if use_conv:
            self.down = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            assert (
                self.in_channels == self.out_channels
            ), "If not using conv, in and out channels must be the same"
            self.down = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, style=None):
        x = self.down(x)
        return x


class ResBlockDown(nn.Module):
    def __init__(self, in_chan, out_chan, padding, padding_mode):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.padding = padding
        self.padding_mode = padding_mode

        self.skip = nn.Conv3d(in_chan, out_chan, kernel_size=1, padding=0)
        self.op_skip = Downsample(out_chan, use_conv=False)
        self.main_block = nn.ModuleList(
            [
                nn.SiLU(),
                nn.Conv3d(
                    in_chan,
                    out_chan,
                    kernel_size=3,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.SiLU(),
                nn.Conv3d(
                    out_chan,
                    out_chan,
                    kernel_size=3,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
            ]
        )
        self.op_main = Downsample(in_chan, use_conv=False)

    def forward(self, x):
        x_skip = self.skip(x)
        x_skip = self.op_skip(x_skip)
        for op in self.main_block:
            x = op(x)
        x = self.op_main(x)
        x = x + x_skip
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size=1,
        base_chan=32,
        padding=1,
        padding_mode="zeros",
        eul_scale_factor=2,
        input_shape=128,
        use_inv_shuffle=True,
        cat_eul=True,
        use_fourier=False,
        **kwargs,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.style_size = style_size
        self.base_chan = base_chan
        self.padding = padding
        self.padding_mode = padding_mode
        self.eul_scale_factor = eul_scale_factor
        self.use_inv_shuffle = use_inv_shuffle
        self.cat_eul = cat_eul
        self.cat_dens = False
        self.use_fourier = use_fourier

        self.lag2eul = Lagrangian2Eulerian(eul_scale_factor=eul_scale_factor)

        self.head = nn.Conv3d(
            in_chan + eul_scale_factor**3,
            base_chan,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.resdown1 = ResBlockDown(base_chan, base_chan * 2, padding, padding_mode)
        self.resdown2 = ResBlockDown(
            base_chan * 2, base_chan * 4, padding, padding_mode
        )

        self.resdown3 = ResBlockDown(
            base_chan * 4, base_chan * 8, padding, padding_mode
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(
                base_chan * 8,
                base_chan * 8,
                kernel_size=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.SiLU(),
            nn.Conv3d(
                base_chan * 8,
                1,
                kernel_size=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x, style, cond=None):
        eul = self.lag2eul(x[:, :3], a=np.float64(style))
        eul = pixel_shuffle_3d_inv(eul, self.eul_scale_factor)
        x = torch.cat([eul, x], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        x = self.head(x)
        x = self.resdown1(x)
        x = self.resdown2(x)
        x = self.resdown3(x)
        x = self.out(x)
        return x


class Lagrangian2Eulerian(nn.Module):
    def __init__(
        self,
        eul_scale_factor=2,
        eul_pad=0,
        rm_dis_mean=True,
        periodic=False,
        dis_std=6.0,  # in Mpc/h
        boxsize=100.0,  # in Mpc/h
        meshsize=512,
        inv_shuffle=False,
        **kwargs,
    ):
        super().__init__()
        self.eul_scale_factor = eul_scale_factor
        self.eul_pad = eul_pad
        self.rm_dis_mean = rm_dis_mean
        self.periodic = periodic
        self.dis_norm_coeff = dis_std * meshsize / boxsize * eul_scale_factor
        self.inv_shuffle = inv_shuffle

    def forward(self, dis, val=1.0, a=1.0):
        eul_scale_factor = self.eul_scale_factor
        eul_pad = self.eul_pad

        dis_norm = self.dis_norm_coeff * D_growth(a)
        if self.rm_dis_mean:
            dis_mean = sum(dis.mean((2, 3, 4), keepdim=True))

        dtype, device = dis.dtype, dis.device
        N, DHW = dis.shape[0], dis.shape[2:]

        DHW = torch.Size([s * eul_scale_factor + 2 * eul_pad for s in DHW])
        mesh = torch.zeros(N, 1, *DHW, dtype=dtype, device=device)

        pos = (dis - dis_mean) * dis_norm
        del dis

        pos[:, 0] += torch.arange(
            0.5, DHW[0] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device
        )[:, None, None]
        pos[:, 1] += torch.arange(
            0.5, DHW[1] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device
        )[:, None]
        pos[:, 2] += torch.arange(
            0.5, DHW[2] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device
        )

        pos = pos.contiguous().view(N, 3, -1, 1)

        intpos = pos.floor().to(torch.int)
        neighbors = (
            torch.arange(8, device=device)
            >> torch.arange(3, device=device)[:, None, None]
        ) & 1

        tgtpos = intpos + neighbors
        del intpos, neighbors

        kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)

        val = val * kernel
        del kernel

        tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
        val = val.view(N, 1, -1)

        bounds = torch.tensor(DHW, device=device)[:, None]

        for n in range(N):  # because ind has variable length
            bounds = torch.tensor(DHW, device=device)[:, None]

            if self.periodic:
                torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

            ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]) * DHW[2] + tgtpos[n, 2]
            src = val[n]

            if not self.periodic:
                mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                ind = ind[mask]
                src = src[:, mask]

            mesh[n].view(1, -1).index_add_(1, ind, src)

        return mesh


def pixel_shuffle_3d_inv(x, r):
    """
    Rearranges tensor x with shape ``[B,C,H,W,D]``
    to a tensor of shape ``[B,C*r*r*r,H/r,W/r,D/r]``.
    """
    [B, C, H, W, D] = list(x.size())
    x = x.contiguous().view(B, C, H // r, r, W // r, r, D // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.contiguous().view(B, C * (r**3), H // r, W // r, D // r)
    return x


def D_growth(a, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero"""
    OL = 1 - Om
    return (
        a
        * hyp2f1(1, 1 / 3, 11 / 6, -OL * a**3 / Om)
        / hyp2f1(1, 1 / 3, 11 / 6, -OL / Om)
    )
