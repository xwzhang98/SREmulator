from math import log2
import torch
import torch.nn as nn
from functools import partial

from .narrow import narrow_by
from .resample import Resampler
from .style import ConvStyled3d


class LeakyReLUStyled(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, style=None):
        return super().forward(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        scale_factor=8,
        chan_base=512,
        chan_min=64,
        chan_max=512,
        **kwargs
    ):
        """The StyleGAN2 generator.

        Args:
            in_chan (torch.Tensor): input channel, on default 3+3 for displacement and velocity
            out_chan (torch.Tensor): output channel, on default 3+3 for displacement and velocity
            style_size (_type_): dimension of the style vector, on default (1,1)
            scale_factor (int, optional): upscaling factor. Defaults to 8.
            chan_base (int, optional): base channel number. Defaults to 512.
            chan_min (int, optional): minimum channel number. Defaults to 64.
            chan_max (int, optional): maximum channel number. Defaults to 512.

        Returns:
            y: super-resolution output
        """
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))
        self.num_blocks = num_blocks

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.shallow = ConvStyled3d(in_chan, chan(0), self.style_size, 1)
        self.act = LeakyReLUStyled(0.2, True)

        self.initcond_conv = ConvStyled3d(in_chan, out_chan, self.style_size, 3)

        hblock_channels = [[chan(i), chan(i + 1)] for i in range(num_blocks)]

        hblock = partial(
            HBlock, in_chan=in_chan, out_chan=out_chan, style_size=style_size
        )

        self.hblock0 = hblock(*hblock_channels[0])
        self.hblock1 = hblock(*hblock_channels[1])
        self.hblock2 = hblock(*hblock_channels[2])

    def forward(
        self,
        x: torch.Tensor,  # size (batch_size, 6, 16, 16, 16) pad 3
        style: torch.Tensor,  # size (1,1)
        z: torch.Tensor,  # size (batch_size, 6, 128, 128, 128) pad 4
    ):

        # shallow feature extraction
        x = self.shallow((x, style))
        x = self.act(x)

        # conv layer for initcond
        z = self.initcond_conv((z, style))

        # start HBlocks
        x, y = self.hblock0(x, y, style)
        x, y = self.hblock1(x, y, style)
        x, y = self.hblock2(x, y, style)

        # learn difference from initcond to final output
        z_shape = z.shape[-1]
        y_shape = y.shape[-1]
        if z_shape != y_shape:
            narrow_edge = (z_shape - y_shape) // 2
            z = narrow_by(z, narrow_edge)

        z += y

        return z


class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, in_chan, out_chan, style_size):
        super().__init__()

        self.act = LeakyReLUStyled(0.2, True)

        self.upsample = Resampler(3, 2)

        self.conv1 = ConvStyled3d(prev_chan, next_chan, style_size, 3)

        self.conv2 = ConvStyled3d(next_chan, next_chan, style_size, 3)

    def forward(self, x, y, s):
        # ------------ Left branch ------------
        x = self.upsample(x)

        # first styled conv, same as conv in styled_srsgan
        x = self.conv1((x, s))
        x = self.act(x)

        # second styled block, same as conv1 in styled_srsgan
        x = self.conv2((x, s))
        x = self.act(x)

        # ------------ Right branch(RGB branch) ------------
        # direct upsampling
        y = self.upsample(y)

        # crop y to match x
        x_size = x.shape[-1]
        y_size = y.shape[-1]
        if x_size != y_size:
            narrow_edge = (y_size - x_size) // 2
            y = narrow_by(y, narrow_edge)

        # To RGB block, same as proj in styled_srsgan
        feature_map = self.proj((x, s))
        # add feature map to y
        y = y + feature_map

        return x, y
