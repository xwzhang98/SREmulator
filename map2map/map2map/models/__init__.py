from .adversary import grad_penalty_reg
from .conv import ConvBlock, ResBlock
from .instance_noise import InstanceNoise
from .lag2eul import *
from .narrow import narrow_by, narrow_cast, narrow_like
from .patchgan import PatchGAN, PatchGAN42
from .power import power
from .resample import resample, Resampler, Resampler2
from .spectral_norm import add_spectral_norm, rm_spectral_norm

from .style import *
from .styled_conv import *
from .styled_srsgan import *
from .simple_unet import *
from .discriminator import *


from .unet import *
from .vnet import VNet


from .wasserstein import WDistLoss, wasserstein_distance_loss, wgan_grad_penalty


from .instance_noise import InstanceNoise
