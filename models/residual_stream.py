import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

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

# This is the original implementation, and follows the nGPT paper
class ResidualSphericalLERP(Module):
    def __init__(
        self,
        fn: Module,
        dim: int,
        init: float,
        scale: float | None = None,
        groups = 1,
        norm_eps = 0.
    ):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim ** -0.5))
        self.l2norm = L2Norm(dim = -1, norm_eps = norm_eps, groups = groups)

    def forward(self, x, **kwargs):
        residual = x

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        out = self.l2norm(out)
        out = self.l2norm(residual.lerp(out, self.branch_scale()))

        if tuple_output:
            out = (out, *rest)

        return out


# TODO / FIXME: consider whether we need a special intialization for the scale parameter like in the original implementation
class ResidualSphericalSLERP(nn.Module):
    """
    This module implements spherical interpolation (slerp) between two vectors on the unit-norm hypersphere.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Unlike HypersphereLERP, this does not use a linear approximation, and strictly enforces alpha to be in [0,1].

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    single_weight : bool, optional
        If True, use a single scalar weight for all dimensions; otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x, y)
        Performs the spherical interpolation on the hypersphere between tensors x and y.
    """

    def __init__(self, fn, dim, single_weight=True):
        super().__init__()

        self.fn = fn
        self.dim = dim
        self.single_weight = single_weight

        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight = nn.Parameter(torch.ones(1) if single_weight else torch.ones(self.dim), requires_grad=True)
        # what is geometric interpretation of single_weight = False?

    def forward(self, x, **kwargs):
        # x, y: [batch_size, ..., dim]

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        # normalize to unit norm
        x, out = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(out, p=2, dim=-1)
        cos_theta = (x * out).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight.sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * out
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        if tuple_output:
            x = (x, *rest)

        return x

class ResidualAdaptiveSphericalSLERP(nn.Module):
    """
    An adaptive variant of HypersphereSLERP, where the interpolation weight is a learned function of the update direction y.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = the angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Here, alpha is computed as alpha = sigmoid(y * W_alpha), where W_alpha is a learnable weight matrix.

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    single_weight : bool, optional
        If True, use a single scalar weight for all dimensions; otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x, y)
        Performs the (adaptive) spherical interpolation on the hypersphere between tensors x and y.
    """

    def __init__(self, fn, dim, single_weight=True):
        super().__init__()

        self.fn = fn
        self.dim = dim
        self.single_weight = single_weight

        # linear map from y to interpolation weight alpha
        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight_map = nn.Linear(dim, 1) if single_weight else nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        # x, y: [batch_size, ..., dim]
        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        # normalize to unit norm
        x, out = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(out, p=2, dim=-1)
        cos_theta = (x * out).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight_map(out).sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * out
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        if tuple_output:
            x = (x, *rest)

        return x
