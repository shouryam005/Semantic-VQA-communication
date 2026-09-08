"""
Complex-valued layers with explicit, correct initialization.

Written rather than imported so that initialization and parameter count are
under our control. The notebook version had two problems:

  nn.Linear(64, 48).to(torch.cfloat)
      Builds a real layer, initializes it with PyTorch's real Kaiming-uniform
      bound, then casts. Every weight starts with a zero imaginary part, and
      the variance is wrong for a complex fan-in. PyTorch also warns that
      complex nn.Module parameters are unsupported.

  torchcvnn / complextorch layers
      Fine in themselves, but their initialization is not visible at the call
      site, which makes it impossible to say the two models were initialized
      comparably.

Initialization here follows Trabelsi et al., "Deep Complex Networks" (ICLR
2018): the weight magnitude is drawn from a Rayleigh distribution with scale
sigma and the phase from Uniform(-pi, pi), giving E[|W|^2] = 2 sigma^2. With
sigma = 1/sqrt(fan_in) this matches the He variance a real ReLU network gets,
so the real and complex models start with comparable activation scale.
"""

import math

import torch
import torch.nn as nn


def complex_init(fan_in, shape, generator=None):
    """Rayleigh magnitude, uniform phase. Returns (real, imag) tensors."""
    sigma = 1.0 / math.sqrt(max(fan_in, 1))
    u = torch.rand(shape, generator=generator).clamp_min(torch.finfo(torch.float32).tiny)
    magnitude = sigma * torch.sqrt(-2.0 * torch.log(u))
    phase = torch.empty(shape).uniform_(-math.pi, math.pi, generator=generator)
    return magnitude * torch.cos(phase), magnitude * torch.sin(phase)


class ComplexConv2d(nn.Module):
    """
    Complex convolution via the Gauss form of complex multiplication.

    For X = Xr + jXi and W = Wr + jWi,

        Re(W * X) = Wr*Xr - Wi*Xi
        Im(W * X) = Wr*Xi + Wi*Xr

    Bias is a single complex vector applied once to the output, not two real
    biases applied inside the four sub-convolutions -- which is what happens if
    you build this out of two nn.Conv2d layers that each carry their own bias.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = in_channels * kernel_size * kernel_size

        weight_r, weight_i = complex_init(fan_in, shape)
        self.weight_r = nn.Parameter(weight_r)
        self.weight_i = nn.Parameter(weight_i)

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels))
            self.bias_i = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        self.padding = padding
        self.stride = stride

    def forward(self, x):
        xr, xi = x.real, x.imag
        conv = lambda inp, w: nn.functional.conv2d(inp, w, None, self.stride, self.padding)

        real = conv(xr, self.weight_r) - conv(xi, self.weight_i)
        imag = conv(xi, self.weight_r) + conv(xr, self.weight_i)

        if self.bias_r is not None:
            real = real + self.bias_r.view(1, -1, 1, 1)
            imag = imag + self.bias_i.view(1, -1, 1, 1)

        return torch.complex(real, imag)


class ComplexLinear(nn.Module):
    """Complex fully-connected layer, same multiplication rule as above."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        weight_r, weight_i = complex_init(in_features, (out_features, in_features))
        self.weight_r = nn.Parameter(weight_r)
        self.weight_i = nn.Parameter(weight_i)

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    def forward(self, x):
        xr, xi = x.real, x.imag
        real = nn.functional.linear(xr, self.weight_r) - nn.functional.linear(xi, self.weight_i)
        imag = nn.functional.linear(xi, self.weight_r) + nn.functional.linear(xr, self.weight_i)

        if self.bias_r is not None:
            real = real + self.bias_r
            imag = imag + self.bias_i

        return torch.complex(real, imag)


class CReLU(nn.Module):
    """ReLU applied separately to the real and imaginary parts."""

    def forward(self, x):
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))


class ModReLU(nn.Module):
    """
    ReLU on the modulus, phase preserved: relu(|z| + b) * z/|z|.

    Included as an alternative to CReLU. CReLU is not phase-equivariant -- it
    carves the complex plane into quadrants and collapses three of them --
    which is one candidate explanation for a complex model that fails to use
    phase. ModReLU leaves phase untouched.
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        modulus = torch.abs(x)
        shape = [1, -1] + [1] * (x.dim() - 2)
        scale = torch.relu(modulus + self.bias.view(shape)) / (modulus + self.eps)
        return x * scale.to(x.dtype)


class ComplexMaxPool2d(nn.Module):
    """
    Max pooling that selects by modulus and keeps the winning complex value.

    Ordering complex numbers is not defined; taking the max of the real and
    imaginary parts independently would select a value that is not present in
    the input and would corrupt phase. This selects the entry of largest
    magnitude and returns it intact.
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        _, indices = nn.functional.max_pool2d(
            torch.abs(x), self.kernel_size, self.stride, return_indices=True
        )
        n, c, h, w = indices.shape
        flat = lambda t: t.flatten(2).gather(2, indices.flatten(2)).view(n, c, h, w)
        return torch.complex(flat(x.real), flat(x.imag))


class ComplexAdaptiveAvgPool2d(nn.Module):
    """Averaging is linear, so it applies to the real and imaginary parts."""

    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        pool = lambda t: nn.functional.adaptive_avg_pool2d(t, self.output_size)
        return torch.complex(pool(x.real), pool(x.imag))


class ZReLU(nn.Module):
    """
    Identity inside the first quadrant, zero elsewhere (Guberman 2016).

    Unlike CReLU it never moves a value's phase -- it either keeps the number
    exactly or discards it -- so surviving activations carry undistorted phase.
    """

    def forward(self, x):
        keep = (x.real >= 0) & (x.imag >= 0)
        return x * keep.to(x.dtype)


class Cardioid(nn.Module):
    """
    Phase-preserving smooth activation (Virtue et al. 2017):

        f(z) = 0.5 * (1 + cos(arg z)) * z

    The gain depends only on phase and the phase itself passes through
    untouched, so this is the complex analogue of ReLU that a real network
    gets for free: it gates magnitude without rotating anything.
    """

    def forward(self, x):
        gain = 0.5 * (1.0 + torch.cos(torch.angle(x)))
        return x * gain.to(x.dtype)


ACTIVATIONS = {"crelu": CReLU, "modrelu": ModReLU, "zrelu": ZReLU, "cardioid": Cardioid}


def make_activation(name, num_features):
    """ModReLU carries a per-channel bias; the others are parameter-free."""
    if name == "modrelu":
        return ModReLU(num_features)
    return ACTIVATIONS[name]()
