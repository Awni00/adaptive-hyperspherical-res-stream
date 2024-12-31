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


# TODO: consider whether we need a special intialization for the scale parameter like in ResidualSphericalLERP
# currently, the scale parameter is initialized to 0, such that it comptues a balanced interpolation between the residual and the update
class ResidualSphericalSLERP(nn.Module):
    """
    This module implements spherical interpolation (slerp) between two vectors on the unit-norm hypersphere.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Unlike HypersphereLERP, this does not use a linear approximation, and strictly enforces alpha to be in [0,1].

    Optionally, if n_spheres > 1, the vectors are factored into n_spheres, normalized and interpolated along each sphere independently.

    Parameters
    ----------
    fn: nn.Module
        The function that generates the output tensor.
    dim : int
        Dimension of the input and output tensors.
    n_spheres: int, optional
        Number of spheres to factorize the dimension into. Will interpolate across each sphere independently. Default is 1.
    single_weight : bool, optional
        If True, use a single scalar interpolation weight (per sphere); otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x, y)
        Performs the spherical interpolation on the hypersphere between tensors x and y.
    """

    def __init__(self, fn, dim, n_spheres=1, single_weight=True, slerp_weight_init=0.0):
        super().__init__()

        self.fn = fn
        self.dim = dim
        self.n_spheres = n_spheres # number of spheres to factorize the dimension into
        # if n_spheres > 1, then interpolate along each sphere independently
        # in this case, the output will be unit-norm along each sphere ("factor" of dim), but the full vector will not be unit-norm

        self.sphere_dim = dim // n_spheres
        assert dim % n_spheres == 0, 'Dimension must be divisible by n_spheres'
        self.single_weight = single_weight
        self.slerp_weight_init = slerp_weight_init

        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight = nn.Parameter(
            self.slerp_weight_init*torch.ones(self.n_spheres) if single_weight else self.slerp_weight_init*torch.ones(self.dim),
            requires_grad=True)

        # note: single_weight=True interpolates between two vectors on the unit-norm hypersphere (if n_sphere=1)
        # or factors the dimension into n_spheres and interpolates along each sphere independently (if n_spheres>1)
        # single_weight=False loses some of the geometric interpolation, performing a more complex "interpolation"
        # where the weight along each dimension (in each sphere) is independently parameterized.
        # This process itself does not preserve unit-norm, but the output is normalized again at the end.

    def forward(self, x, **kwargs):
        # x: [batch_size, ..., dim]
        # kwargs: additional arguments to pass to the function

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        # factor into n_spheres and normalizes each sphere
        x = self.factor_and_normalize_spheres(x) # shape: [batch_size, ..., n_spheres, d_sphere]
        out = self.factor_and_normalize_spheres(out) # shape: [batch_size, ..., n_spheres, d_sphere]

        x = self.interpolate_factored_spheres(x, out, self.slerp_weight) # shape: [batch_size, ..., n_spheres, d_sphere]

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            # normalize to unit norm, along each spehre
            x = torch.nn.functional.normalize(x, p=2, dim=-1) # shape: [batch_size, ..., n_spheres, d_sphere]

        # note: x is not unit-norm along the full dimension, but is unit-norm along each sphere

        # rearrange to re-pack spheres together into single vector
        x = rearrange(x, '... ns ds -> ... (ns ds)')

        if tuple_output:
            x = (x, *rest)

        return x

    def factor_and_normalize_spheres(self, x):
        # x: [batch_size, ..., dim]
        # n_spheres: int, number of spheres to factorize the dimension into

        # factorize the dimension into n_spheres (i.e., [1, ..., dim/n_spheres], [dim/n_spheres+1, ..., 2*dim/n_spheres], ...)
        # and normalize each factorized sphere

        factored_spheres = rearrange(x, '... (ns ds) -> ... ns ds', ns=self.n_spheres, ds=self.sphere_dim)

        factored_spheres = torch.nn.functional.normalize(factored_spheres, p=2, dim=-1)

        return factored_spheres

    def interpolate_factored_spheres(self, x, y, slerp_weights):
        # x, y: [batch_size, ..., n_spheres, d_sphere]
        # slerp_weights: [n_spheres] or [dim], interpolation weights for each sphere or dimension

        # factorize the dimension into n_spheres (i.e., [1, ..., dim/n_spheres], [dim/n_spheres+1, ..., 2*dim/n_spheres], ...)
        # and interpolate between the factored spheres

        # compute angle between x and y along each sphere
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., n_spheres, 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., n_spheres, 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., n_spheres, 1]

        alpha = slerp_weights.sigmoid() # shape: [n_spheres] or [dim]
        if self.single_weight:
            alpha = alpha.unsqueeze(-1) # shape: [n_spheres, 1]
        else:
            alpha = rearrange(alpha, '(ns ds) -> ns ds', ns=self.n_spheres) # shape: [n_spheres, d_sphere]

        coef_x = torch.sin((1 - alpha) * theta) / sin_theta # shape: [batch_size, ..., n_spheres, 1] or [batch_size, ..., n_spheres, d_sphere]
        coef_y = torch.sin(alpha * theta) / sin_theta # shape: [batch_size, ..., n_spheres, 1] or [batch_size, ..., n_spheres, d_sphere]

        out = coef_x * x + coef_y * y
        # out.shape: [batch_size, ..., n_spheres, d_sphere] s.t. norm(out, dim=-1) = 1

        return out

class ResidualAdaptiveSphericalSLERP(nn.Module):
    """
    An adaptive variant of HypersphereSLERP, where the interpolation weight is a learned function of the update direction y.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = the angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Here, alpha is computed as alpha = sigmoid(y * W_alpha), where W_alpha is a learnable weight matrix.

    Optionally, if n_spheres > 1, the vectors are factored into n_spheres, normalized and interpolated along each sphere independently.

    Parameters
    ----------
    fn: nn.Module
        The function that generates the output tensor.
    dim : int
        Dimension of the input and output tensors.
    n_spheres: int, optional
        Number of spheres to factorize the dimension into. Will interpolate across each sphere independently. Default is 1.
    single_weight : bool, optional
        If True, use a single scalar interpolation weight (per sphere); otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x)
        Performs the spherical interpolation on the hypersphere between tensors x and y = self.fn(x).
    """

    def __init__(self, fn, dim, n_spheres=1, single_weight=True, slerp_weight_init=0.0):
        super().__init__()

        self.fn = fn
        self.dim = dim
        self.n_spheres = n_spheres # number of spheres to factorize the dimension into
        # if n_spheres > 1, then interpolate along each sphere independently
        # in this case, the output will be unit-norm along each sphere ("factor" of dim), but the full vector will not be unit-norm

        self.sphere_dim = dim // n_spheres
        assert dim % n_spheres == 0, 'Dimension must be divisible by n_spheres'
        self.single_weight = single_weight
        self.slerp_weight_init = slerp_weight_init

        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight_map = nn.Linear(dim, self.n_spheres) if single_weight else nn.Linear(dim, dim)

        # note: single_weight=True interpolates between two vectors on the unit-norm hypersphere (if n_sphere=1)
        # or factors the dimension into n_spheres and interpolates along each sphere independently (if n_spheres>1)
        # single_weight=False loses some of the geometric interpolation, performing a more complex "interpolation"
        # where the weight along each dimension (in each sphere) is independently parameterized.
        # This process itself does not preserve unit-norm, but the output is normalized again at the end.

    def forward(self, x, **kwargs):
        # x: [batch_size, ..., dim]
        # kwargs: additional arguments to pass to the function

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        # compute interpolation weights
        # NOTE: here, we've chosen to compute the interpolation weights based on the output before normalization or factorization
        # TODO: it may be reasonable to compute the weight for the i-th sphere based on the normalized i-th sphere only
        # similarly, it may be beneficial to use NormLinear instead of Linear
        slerp_weight = self.slerp_weight_map(out) # shape: [batch_size, ..., n_spheres]

        # factor into n_spheres and normalizes each sphere
        x = self.factor_and_normalize_spheres(x) # shape: [batch_size, ..., n_spheres, d_sphere]
        out = self.factor_and_normalize_spheres(out) # shape: [batch_size, ..., n_spheres, d_sphere]


        x = self.interpolate_factored_spheres(x, out, slerp_weight) # shape: [batch_size, ..., n_spheres, d_sphere]

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            # normalize to unit norm, along each spehre
            x = torch.nn.functional.normalize(x, p=2, dim=-1) # shape: [batch_size, ..., n_spheres, d_sphere]

        # note: x is not unit-norm along the full dimension, but is unit-norm along each sphere

        # rearrange to re-pack spheres together into single vector
        x = rearrange(x, '... ns ds -> ... (ns ds)')

        if tuple_output:
            x = (x, *rest)

        return x

    def factor_and_normalize_spheres(self, x):
        # x: [batch_size, ..., dim]
        # n_spheres: int, number of spheres to factorize the dimension into

        # factorize the dimension into n_spheres (i.e., [1, ..., dim/n_spheres], [dim/n_spheres+1, ..., 2*dim/n_spheres], ...)
        # and normalize each factorized sphere

        factored_spheres = rearrange(x, '... (ns ds) -> ... ns ds', ns=self.n_spheres, ds=self.sphere_dim)

        factored_spheres = torch.nn.functional.normalize(factored_spheres, p=2, dim=-1)

        return factored_spheres

    def interpolate_factored_spheres(self, x, y, slerp_weights):
        # x, y: [batch_size, ..., n_spheres, d_sphere]
        # slerp_weights: [n_spheres] or [dim], interpolation weights for each sphere or dimension

        # factorize the dimension into n_spheres (i.e., [1, ..., dim/n_spheres], [dim/n_spheres+1, ..., 2*dim/n_spheres], ...)
        # and interpolate between the factored spheres

        # compute angle between x and y along each sphere
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., n_spheres, 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., n_spheres, 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., n_spheres, 1]

        alpha = slerp_weights.sigmoid() # shape: [batch_size, ..., n_spheres] or [batch_size, ..., dim]
        if self.single_weight:
            alpha = alpha.unsqueeze(-1) # shape: [batch_size, ..., n_spheres, 1]
        else:
            alpha = rearrange(alpha, '... (ns ds) -> ... ns ds', ns=self.n_spheres) # shape: [batch_size, ..., n_spheres, sphere_dim]

        coef_x = torch.sin((1 - alpha) * theta) / sin_theta # shape: [batch_size, ..., n_spheres, 1] or [batch_size, ..., n_spheres, sphere_dim]
        coef_y = torch.sin(alpha * theta) / sin_theta # shape: [batch_size, ..., n_spheres, 1] or [batch_size, ..., n_spheres, sphere_dim]

        out = coef_x * x + coef_y * y
        # out.shape: [batch_size, ..., n_spheres, d_sphere] s.t. norm(out, dim=-1) = 1 (if single_weight=True)

        return out