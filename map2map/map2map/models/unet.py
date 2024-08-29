import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from scipy.special import hyp2f1


def normalization(channels):
    return nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def time_embedding(
    a: torch.Tensor,
    embedding_dim: int,
):
    """
    Compute the style embedding for a given scale factor.

    Args:
        a (torch.Tensor): Scale factor, range from 0 to 1.
        embedding_dim (int): Embedding dimension.

    Returns:
        torch.Tensor: Style embedding.
    """
    # a (N, 1)
    # a *= 1000
    half = embedding_dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=a.device)
        * (-math.log(10000.0) / half)
    )  # size (half)
    args = a * 1000 * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def narrow_as(x, y):
    """
    _summary_

    Args:
        x (tensor): tensor to narrow, shape must be larger than y
        y (tensor): tensor to narrow to

    Returns:
        _type_: new x that is narrow to y
    """
    # x N C D H W
    # y N C d h w
    if x.size() == y.size():
        return x
    else:
        edge = (x.size()[-1] - y.size()[-1]) // 2
        for d in range(2, x.dim()):
            x = x.narrow(d, edge, x.size()[d] - 2 * edge)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, a):
        return time_embedding(a, self.embedding_dim)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        use_conv,
        out_channels=None,
        padding=0,
        padding_mode="zeros",
    ):
        """a basic upsampling module

        Args:
            in_channels (int): input channels
            use_conv (bool): whether to use a modulated conv or a normal conv after upsampling
            style_size (Optional[int], optional): style input size. Defaults to None.
            out_channels (Optional[int], optional): output channels. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.padding = padding
        if use_conv:
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            )

    def forward(self, x, style=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

    @torch.no_grad()
    def _calc_out_shape(self, in_shape):
        assert len(in_shape) == 5, "Input shape must be 5D"
        in_shape[1] = self.out_channels
        for i in range(2, 5):
            if self.use_conv:
                in_shape[i] = in_shape[i] * 2 - 2 * self.padding
            else:
                in_shape[i] = in_shape[i] * 2
        return in_shape


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

    @torch.no_grad()
    def _calc_out_shape(self, in_shape):
        assert len(in_shape) == 5, "Input shape must be 5D"
        in_shape[1] = self.out_channels
        for i in range(2, 5):
            in_shape[i] = math.ceil(in_shape[i] / 2)
        return in_shape


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.SiLU(),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_channels = embedding_channels
        self.padding = padding

        self.in_layers = nn.ModuleList(
            [
                normalization(in_channels),
                nn.SiLU(),
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
            ]
        )

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * out_channels),
        )

        self.out_layers = nn.ModuleList(
            [
                normalization(out_channels),
                nn.SiLU(),
                zero_module(
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                ),
            ]
        )

        if self.in_channels != self.out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    @torch.no_grad()
    def _calc_out_shape(self, in_shape):
        assert len(in_shape) == 5, "Input shape must be 5D"
        out_channels = self.out_channels
        in_shape[1] = out_channels
        kernel_size = 3
        for i in range(2, 5):
            # start in layers
            in_shape[i] = in_shape[i] - kernel_size + 1 + 2 * self.padding
            in_shape[i] = in_shape[i] - kernel_size + 1 + 2 * self.padding  # 2 convs

        return in_shape

    def forward(self, x, style):
        """
        _summary_

        Args:
            x (_type_): _description_
            style (_type_): _description_

        Returns:
            _type_: _description_
        """
        # embedding
        s = self.embed_layers(style)
        s = s.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # skip branch
        skip = self.skip(x)
        # main branch
        for layer in self.in_layers:
            x = layer(x)

        norm, rest = self.out_layers[0], self.out_layers[1:]
        scale, bias = torch.chunk(s, 2, dim=1)
        x = norm(x) * (1 + scale) + bias
        for layer in rest:
            x = layer(x)

        skip = narrow_as(skip, x)
        x = x + skip

        return x


class ResidualUpDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        padding: int = 0,
        padding_mode: str = "zeros",
        up=False,
        down=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_channels = embedding_channels
        self.padding = padding
        self.up = up
        self.down = down

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * out_channels),
        )

        if up and not down:
            self.op_skip = Upsample(
                in_channels=in_channels,
                out_channels=out_channels,
                use_conv=False,
                padding=padding,
                padding_mode=padding_mode,
            )
            self.op_main = Upsample(
                in_channels=in_channels,
                out_channels=out_channels,
                use_conv=False,
                padding=padding,
                padding_mode=padding_mode,
            )
        elif down and not up:
            self.op_skip = Downsample(
                in_channels=in_channels,
                out_channels=out_channels,
                use_conv=False,
                padding=padding,
                padding_mode=padding_mode,
            )
            self.op_main = Downsample(
                in_channels=in_channels,
                out_channels=out_channels,
                use_conv=False,
                padding=padding,
                padding_mode=padding_mode,
            )
        else:
            raise ValueError("Must specify either up or down")
        self.in_layers = nn.ModuleList(
            [
                normalization(in_channels),
                nn.SiLU(),
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                normalization(out_channels),
                nn.SiLU(),
                zero_module(
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                ),
            ]
        )

        if self.in_channels != self.out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    @torch.no_grad()
    def _calc_out_shape(self, in_shape):
        # in_shape N C D H W
        assert len(in_shape) == 5, "Input shape must be 5D"
        out_channels = self.out_channels
        in_shape[1] = out_channels
        kernel_size = 3
        for i in range(2, 5):
            # start in layers
            if self.up:
                in_shape[i] = in_shape[i] * 2
            elif self.down:
                in_shape[i] = math.ceil(in_shape[i] / 2)
            in_shape[i] = in_shape[i] - kernel_size + 1 + 2 * self.padding  # conv
            in_shape[i] = in_shape[i] - kernel_size + 1 + 2 * self.padding  # conv

        return in_shape

    def forward(self, x, style):
        # embedding
        s = self.embed_layers(style)
        s = s.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # skip branch
        skip = self.op_skip(x)
        skip = self.skip(skip)
        # main branch
        before_conv, conv = self.in_layers[:-1], self.in_layers[-1]
        for layer in before_conv:
            x = layer(x)
        x = self.op_main(x)
        x = conv(x)

        norm, rest = self.out_layers[0], self.out_layers[1:]
        scale, bias = torch.chunk(s, 2, dim=1)
        x = norm(x) * (1 + scale) + bias
        for layer in rest:
            x = layer(x)

        skip = narrow_as(skip, x)
        x = x + skip

        return x


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        """
        b, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        qkv = rearrange(
            qkv,
            "b (qkvs head ch) len -> b len qkvs head ch",
            qkvs=3,
            head=self.n_heads,
        )
        q, k, v = (
            qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        )  #  b len qkvs head ch -> b head qkvs len ch -> b head qkvs dim len -> b head 1 ch len
        q = q.reshape(b * self.n_heads, ch, length)
        k = k.reshape(b * self.n_heads, ch, length)
        v = v.reshape(b * self.n_heads, ch, length)
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(
            math.sqrt(ch)
        )  # why two sqrt insted of one after calculating weight ?
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = F.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(b, -1, length)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        n_heads,
        dropout=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = self.channels // n_heads
        self.attn = nn.MultiheadAttention(
            channels,
            n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, qkv):
        """
        Apply QKV attention.
        """
        N, _, length = qkv.shape
        qkv = rearrange(
            qkv,
            "b (three embed_dim) len -> b len three embed_dim",
            embed_dim=self.channels,
        )
        # target shape q: (batch_size, seqlen, embed_dim)
        q, k, v = qkv.chunk(3, dim=2)  # b len three embed_dim -> b len embed_dim
        q = q.view(N, length, -1)
        k = k.view(N, length, -1)
        v = v.view(N, length, -1)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            attn, _ = self.attn(
                q, k, v, need_weights=False
            )  # attn: (batch_size, seqlen, nheads, headdim).

        return attn.reshape(N, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        attn_type="qkv",
    ):
        super().__init__()
        self.channels = channels
        self.attn_type = attn_type
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), "Number of head channels must divide number of channels"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1)
        if self.attn_type == "qkv":
            self.attn = QKVAttention(self.num_heads)
        elif self.attn_type == "multihead":
            self.attn = MultiHeadAttention(channels, self.num_heads)
        else:
            raise NotImplementedError(f"Attention type {self.attn_type} not supported")
        self.out = zero_module(nn.Conv3d(channels, channels, kernel_size=1))

    def forward(self, x, s=None):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        N, C, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(N, -1, np.prod(spatial))
        y = self.attn(qkv)
        y = y.view(N, C, *spatial)
        y = self.out(y)
        x = x + y
        return x

    @torch.no_grad()
    def _calc_out_shape(self, in_shape):
        return in_shape


class NaiveBlockWrapper(nn.Sequential):
    def forward(self, x, s=None):
        for module in self:
            x = module(x, s)
        return x


class FourierFeatures(nn.Module):
    def __init__(self, first=7.0, last=8.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        # x shape B, C, D, H, W
        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * torch.pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, 1, 1)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, C, D, H, W)
        features = features.flatten(1, 2)  # (B, F * C, D, H, W)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, D, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        base_chan=64,
        style_size=1,
        channel_mul=[1, 1, 2, 2],
        num_heads=4,
        num_head_channels=-1,
        num_res_blocks=1,
        padding=1,
        padding_mode="zeros",
        conv_updown=True,
        attn_resolutions=[16, 8],
        input_shape=128,
        bypass=True,
        use_checkpoint=False,
        use_fourier_features=False,
        **kwargs,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.base_chan = base_chan
        self.style_size = style_size
        self.style_embed_size = 2 * base_chan
        self.padding = padding
        self.conv_updown = conv_updown
        self.bypass = bypass
        self.use_checkpoint = use_checkpoint
        self.use_fourier_features = use_fourier_features

        self.time_embed = nn.Sequential(
            nn.Linear(base_chan, self.style_embed_size),
            nn.SiLU(),
            nn.Linear(self.style_embed_size, self.style_embed_size),
        )

        chan = base_chan * channel_mul[0]
        self.dis2pos = Dis2Pos()
        total_in_chan = in_chan
        if self.use_fourier_features:
            self.fourier_features = FourierFeatures()
            total_in_chan *= self.fourier_features.num_features

        self.head = nn.Conv3d(
            total_in_chan,
            chan,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.in_blocks = nn.ModuleList([])

        feat_chans = [chan]
        curr_res = input_shape
        for level, mult in enumerate(channel_mul):
            for _ in range(num_res_blocks):
                block = [
                    ResidualBlock(
                        chan,
                        base_chan * mult,
                        self.style_embed_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    ),
                ]
                chan = base_chan * mult
                if curr_res in attn_resolutions:
                    block.append(
                        AttentionBlock(
                            chan,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        ),
                    )
                self.in_blocks.append(NaiveBlockWrapper(*block))
                feat_chans.append(chan)
            if level != len(channel_mul) - 1:
                self.in_blocks.append(
                    ResidualUpDownBlock(
                        chan,
                        chan,
                        self.style_embed_size,
                        padding=padding,
                        padding_mode=padding_mode,
                        down=True,
                    )
                    if conv_updown
                    else Downsample(
                        chan,
                        use_conv=True,
                    )
                )
                curr_res //= 2
                feat_chans.append(chan)

        self.mid_blocks = nn.ModuleList([])
        block = [
            ResidualBlock(
                chan,
                chan,
                self.style_embed_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            AttentionBlock(
                chan, num_heads=num_heads, num_head_channels=num_head_channels
            ),
            ResidualBlock(
                chan,
                chan,
                self.style_embed_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
        ]
        self.mid_blocks.append(NaiveBlockWrapper(*block))

        self.out_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mul))[::-1]:
            for _ in range(num_res_blocks + 1):
                f_ch = feat_chans.pop()
                block = [
                    ResidualBlock(
                        chan + f_ch,
                        base_chan * mult,
                        self.style_embed_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                ]
                chan = base_chan * mult
                if curr_res in attn_resolutions:
                    block.append(
                        AttentionBlock(
                            chan,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.out_blocks.append(NaiveBlockWrapper(*block))
            if level != 0:
                self.out_blocks.append(
                    ResidualUpDownBlock(
                        chan,
                        chan,
                        self.style_embed_size,
                        padding=padding,
                        padding_mode=padding_mode,
                        up=True,
                    )
                    if conv_updown
                    else Upsample(chan, use_conv=True)
                )
                curr_res *= 2

        self.out_conv = nn.Sequential(
            normalization(chan),
            nn.SiLU(),
            zero_module(
                nn.Conv3d(
                    chan,
                    out_chan,
                    kernel_size=3,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            ),
        )

    def forward(self, x, style, z=None):
        """
        _summary_

        Args:
            x (_type_): _description_
            style (_type_): _description_
            z (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        # use bypass or not
        if self.bypass:
            x0 = x

        # concate condition with input
        if z is not None:
            x = torch.cat([x, z], dim=1)
        # time embedding
        s = self.time_embed(time_embedding(style, self.base_chan))
        if self.use_fourier_features:
            x = self.fourier_features(x)
        x = self.head(x)

        feat_maps = []
        feat_maps.append(x)
        for layer in self.in_blocks:
            if self.use_checkpoint:
                x = checkpoint(layer, x, s)
            else:
                x = layer(x, s)
            feat_maps.append(x)

        for layer in self.mid_blocks:
            x = layer(x, s)

        for layer in self.out_blocks:
            if isinstance(layer, NaiveBlockWrapper):
                f_map = feat_maps.pop()
                x = torch.cat([x, f_map], dim=1)
                if self.use_checkpoint:
                    x = checkpoint(layer, x, s)
                else:
                    x = layer(x, s)
            else:
                x = layer(x, s)

        x = self.out_conv(x)
        if self.bypass:
            x = x + x0
        return x
