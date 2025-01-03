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
        """
        An L2-Normalized Linear layer.

        Linearly maps from dim to dim_out, such that weights are L2-unit-norm across dim dimension.
        If input is also L2-normalized, the matrix-vector multiplication corresponds to cosine similarities,
        capturing only the relative angle between vectors and not magnitudes.

        Parameters
        ----------
        dim : int
            dimension of input.
        dim_out : int
            dimension of output.
        norm_dim_in : bool, optional
            whether to normalize across `dim` dimension or `dim_out` dimension.
            Default is True, corresponding to normalizing across input dimension.
        parametrize : True, optional
            Whether to register parameterization of weights tensor in linear layer as.
            If normed parameterization is registered, each time linear.weight is accessed, it will be normalized
            and gradients of the backward pass will differentiate through the normalization.
            If True, weights will be initialized as normalized and will continue to be normalized throughout training.
            If False, weights will be initialized as normalized but will *not* be constrained to be normalized after
            being updated through training. By default True.
        norm_eps : float, optional
            epsilon used in L2Norm, by default 0.
        groups : int, optional
            Optionally, chunk/group the `dim` input dimensions and normalize weights across each group.
            By default 1, corresponding to no grouping.
        """

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
