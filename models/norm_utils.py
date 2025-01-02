import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from utils.utils import default, exists, cast_tuple

# scale

class Scale(Module):
    """
    latter part of section 2.5 in the paper
    """
    def __init__(
        self,
        dim,
        init = 1.,
        scale = 1.
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale

# l2 nrom
def l2norm(
    t,
    dim = -1,
    norm_eps = 0.,
    eps = None,
    groups = 1
):
    if groups > 1:
        t = t.chunk(groups, dim = dim)
        t = torch.stack(t)

    if norm_eps == 0.:
        out = F.normalize(t, dim = dim, p = 2)
    else:
        eps = default(eps, 1e-5 if t.dtype == torch.float16 else 1e-10)
        norm = t.norm(dim = dim, keepdim = True)
        target_norm = norm.detach().clamp(min = 1. - norm_eps, max = 1. + norm_eps)
        divisor = norm / target_norm
        out = t / divisor.clamp(min = eps)

    if groups > 1:
        out = torch.cat([*out], dim = dim)

    return out

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1, norm_eps = 0., groups = 1):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.groups = groups

    def forward(self, t):
        return l2norm(t, dim = self.dim, norm_eps = self.norm_eps, groups = self.groups)

# residual slerp update with learned scale

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True,
        parametrize = True,
        norm_eps = 0.,
        groups = 1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        self.scale = groups ** -1
        self.parametrize = parametrize
        self.l2norm = L2Norm(dim = -1 if norm_dim_in else 0, norm_eps = norm_eps, groups = groups)

        if parametrize:
            register_parametrization(
                self.linear,
                'weight',
                self.l2norm
            )

        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original

            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x) * self.scale
