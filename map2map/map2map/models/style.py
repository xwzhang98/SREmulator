import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStyled3d(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        kernel_size=3,
        stride=1,
        bias=True,
        resample=None,
        demodulation=True,
    ):
        super().__init__()

        # self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        # nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
        #                          mode='fan_in', nonlinearity='leaky_relu')
        # self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1
        self.demodulation = demodulation
        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == "U":
            K3 = (2,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = 2
            self.conv = F.conv_transpose3d
        elif resample == "D":
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            raise ValueError("resample type {} not supported".format(resample))
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight,
            a=0.2,
            mode="fan_in",  # effectively 'fan_out' for 'D'
            nonlinearity="leaky_relu",
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        def init_weight(m):
            if type(m) is nn.Linear:
                torch.nn.init.kaiming_uniform_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    torch.nn.init.ones_(m.bias)

        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=in_chan),
        )
        self.style_block.apply(init_weight)

    def forward(self, inputs):
        x, s = inputs[0], inputs[1]
        eps = 1e-8

        N, Cin, *DHWin = x.shape

        C0, C1, *K3 = self.weight.shape

        if self.resample == "U":
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        # s = F.linear(s, self.style_weight, bias=self.style_bias)
        s = self.style_block(s)
        # modulation
        if self.resample == "U":
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s
        # print(Cin, 'Cin2')
        # demodulation
        if self.resample == "U":
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        if self.demodulation:
            w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * C0, C1, *K3)
        # print(N, Cin, *DHWin)
        x = x.reshape(1, N * Cin, *DHWin)
        # END HERE
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape
        # print('N', N, 'Cout', Cout, 'DHWout', *DHWout)
        # x = x.reshape(N, Cout, *DHWout)
        # print(x.shape)
        x = x.view(N, Cout, *DHWout)

        return x


class ModulatedConv3D(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        kernel_size=3,
        stride=1,
        demodulate=True,
        bias=True,
        input_gain=None,
        resample=None,
    ):
        super().__init__()

        self.demodulate = demodulate

        self.resample = resample
        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == "U":
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = 2
            self.conv = F.conv_transpose3d
        elif resample == "D":
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            raise ValueError("resample type {} not supported".format(resample))

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=in_chan)
        )
        self.eps = torch.as_tensor(1e-8)
        self.input_gain = input_gain

    def forward(self, x, style):
        """_summary_

        Args:
            x (_type_): shape (N, Cin, D, H, W)
            style (_type_): shape (N, style_size)
        """
        batch_size, Cin, *DHWin = x.shape

        s = self.style_block(style)  # (N, Cin)

        eps = self.eps  # for numerical stability
        w = self.weight  # (Cout, Cin, K, K, K)

        # prenormalization
        if self.demodulate:
            w = w * w.square().mean([1, 2, 3, 4], keepdim=True).rsqrt()
            s = s * s.square().mean().rsqrt()

        # modulation
        if self.resample != "U":
            s = (
                s.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            )  # (N, 1, Cin, 1, 1, 1)
            w = w.unsqueeze(0)  # (1, Cout, Cin, K, K, K)
            w = w * s  # (N, Cout, Cin, K, K, K)
        else:
            s = (
                s.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            )  # (N, Cin, 1, 1, 1, 1)
            w = w.unsqueeze(0)  # (1, Cin, Cout, K, K, K)
            w = w * s  # (N, Cin, Cout, K, K, K)

        if self.demodulate:
            if self.resample != "U":
                w = w * torch.rsqrt(
                    w.square().sum(dim=[2, 3, 4, 5], keepdim=True) + eps
                )
            else:
                w = w * torch.rsqrt(
                    w.square().sum(dim=[1, 3, 4, 5], keepdim=True) + eps
                )
        # convolution
        x = x.view(1, -1, *x.shape[2:])  # (1, N*Cin, D, H, W)
        w = w.view(-1, *w.shape[2:])  # (N*Cout, Cin, K, K, K)

        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=batch_size)
        x = x.view(batch_size, -1, *x.shape[2:])  # (N, Cout, DHWout)

        return x


class BatchNormStyled3d(nn.BatchNorm3d):
    """Trivially does standard batch normalization, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, s):
        return super().forward(x)


class LeakyReLUStyled(nn.LeakyReLU):
    """Trivially evaluates standard leaky ReLU, but accepts second argument

    for sytle array that is not used
    """

    def forward(self, x, s=None):
        return super().forward(x)


class LeakyReLUStyled2(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for style array that is not used
    
    a copy of LeakyReLUStyled, accept list of inputs
    """

    def forward(self, inputs):
        x = inputs[0]
        return super().forward(x)
