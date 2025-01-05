# Based on: https://github.com/lucidrains/nGPT-pytorch/blob/5208aada3330e366c74ce21f4701f8d2b6aa5761/nGPT_pytorch/nGPT.py

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from .rotary_embedding_torch import RotaryEmbedding
from .residual_stream import ResidualSphericalLERPBase, ResidualAdaptiveSphericalLERP, ResidualSphericalSLERP, ResidualAdaptiveSphericalSLERP
from .norm_utils import NormLinear, Scale, L2Norm, l2norm
from utils.utils import default, exists, cast_tuple

# constants

from torch.nn.attention import SDPBackend

SDP_BACKEND_MAP = dict(
    enable_flash = SDPBackend.FLASH_ATTENTION,
    enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
    enable_math = SDPBackend.MATH,
    enable_cudnn = SDPBackend.CUDNN_ATTENTION
)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        norm_qk = True,
        causal = True,
        manual_norm_weights = False,
        s_qk_init = 1.,
        s_qk_scale = None,
        flash_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True,
            enable_cudnn = True
        ),
        norm_eps = 0.,
        num_hyperspheres = 1,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        # For linear map matrix W: d_model x d_k mapping from d_model to d_k via xW, NormLinear normalizes along d_model-axis
        # this applies also for multi-head vers where W: d_model x (h * d_k)
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights, norm_eps = norm_eps, groups = num_hyperspheres)
        self.l2norm = partial(l2norm, norm_eps = norm_eps, groups = num_hyperspheres)

        dim_sqrt = dim ** 0.5
        self.dim_sqrt = dim_sqrt
        self.attn_scale = dim_head ** 0.5

        dim_inner = dim_head * heads
        self.to_q = NormLinear_(dim, dim_inner)
        self.to_k = NormLinear_(dim, dim_inner)
        self.to_v = NormLinear_(dim, dim_inner)

        # flash attention related context manager

        sdpa_backends = [SDP_BACKEND_MAP[enable_str] for enable_str, enable in flash_kwargs.items() if enable]
        self.sdpa_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)

        # qk rmsnorm + scale

        self.norm_qk = norm_qk
        self.qk_scale = Scale(dim_inner, s_qk_init, default(s_qk_scale, dim ** -1))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear_(dim_inner, dim, norm_dim_in = False)

    def forward(
        self,
        x,
        mask = None,
        rotary_embed: Module | None = None,
        value_residual = None,
        return_values = False
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # maybe value residual, from resformer paper

        if exists(value_residual):
            v = 0.5 * (v + value_residual)

        # rotary positions

        if exists(rotary_embed):
            q = rotary_embed.rotate_queries_or_keys(q)
            k = rotary_embed.rotate_queries_or_keys(k)

        # maybe query key norm

        if self.norm_qk:
            q, k = map(self.l2norm, (q, k))

        # scaling queries and keys - this would line up with the popular use of qk rmsnorm from google deepmind and now black forest labs - will use multihead rmsnorm

        q = q * rearrange(self.qk_scale(), '(h d) -> h 1 d', h = self.heads)

        # for non-autoregressive masking

        if exists(mask):
            row_all_masked_out = ~mask.any(dim = -1)

            mask = rearrange(mask, 'b j -> b 1 1 j')

        # scale is sqrt(dk)

        with self.sdpa_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = self.causal,
                scale = self.attn_scale
            )

        out = self.merge_heads(out)
        out = self.to_out(out)

        if exists(mask) and row_all_masked_out.any():
            out = out.masked_fill(row_all_masked_out[:, None, None], 0.)

        if not return_values:
            return out

        return out, v

# feedforward

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expand_factor = 4,
        manual_norm_weights = False,
        s_hidden_init = 1.,
        s_hidden_scale = 1.,
        s_gate_init = 1.,
        s_gate_scale = 1.,
        norm_eps = 0.,
        num_hyperspheres = 1
    ):
        super().__init__()
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights, norm_eps = norm_eps, groups = num_hyperspheres)

        self.dim = dim
        dim_inner = int(dim * expand_factor * 2 / 3)

        self.to_hidden = NormLinear_(dim, dim_inner)
        self.to_gate = NormLinear_(dim, dim_inner)

        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale)

        self.to_out = NormLinear_(dim_inner, dim, norm_dim_in = False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale()
        gate = gate * self.gate_scale() * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)

# classes

class nGPT(Module):
    def __init__(
        self,
        *,
        vocab_size,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        residual_module = 'ResidualSphericalLERPBase', # SphericalLERP or ResidualAdaptiveSphericalLERP or SphericalSLERP or AdaptiveSphericalSLERP
        residual_module_kwargs = None,
        attn_norm_qk = True,  # they say the query/key normalization is optional
        ff_expand_factor = 4.,
        ce_ignore_index = -1,
        manual_norm_weights = False, # constrain to unit norm through projection after each optimization step, rather than through parameterization (appears to be faster)
        tied_embedding = True,
        num_hyperspheres = 1,
        causal = True,
        add_value_residual = True,
        # below are all the scale related hyperparameters, for controlling effective relative learning rates throughout the network
        alpha_init: float | None = None,  # this would set the alpha init for all residuals, but would be overridden by alpha_attn_init and alpha_ff_init if they are specified
        s_logit_init: float  = 1.,
        s_logit_scale: float | None = None,
        alpha_attn_init: float | tuple[float, ...] | None = None,
        alpha_attn_scale: float | tuple[float, ...] | None = None,
        alpha_ff_init: float | tuple[float, ...] | None = None,
        alpha_ff_scale: float | tuple[float, ...] | None = None,
        s_qk_init: float | tuple[float, ...] = 1.,
        s_qk_scale: float | tuple[float, ...] | None = None,
        s_ff_hidden_init: float | tuple[float, ...] = 1.,
        s_ff_hidden_scale: float | tuple[float, ...] = 1.,
        s_ff_gate_init: float | tuple[float, ...] = 1.,
        s_ff_gate_scale: float | tuple[float, ...] = 1.,
        attn_flash_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        ),
        norm_eps = 0. # greater than 0 allows the norm to be around (1. - norm_eps) to (1. + norm_eps)
    ):
        super().__init__()
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights, norm_eps = norm_eps, groups = num_hyperspheres)
        self.l2norm = partial(l2norm, norm_eps = norm_eps, groups = num_hyperspheres)

        self.dim = dim
        self.causal = causal
        alpha_init = default(alpha_init, 1. / depth)

        self.add_value_residual = add_value_residual # https://arxiv.org/abs/2410.17897v1

        self.token_embed = NormLinear_(dim, vocab_size)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.layers = ModuleList([])

        scale_hparams = (
            alpha_attn_init,
            alpha_attn_scale,
            alpha_ff_init,
            alpha_ff_scale,
            s_qk_init,
            s_qk_scale,
            s_ff_hidden_init,
            s_ff_hidden_scale,
            s_ff_gate_init,
            s_ff_gate_scale
        )

        scale_hparams = tuple(cast_tuple(hparam, depth) for hparam in scale_hparams)

        residual_module_dict = dict(
            ResidualSphericalLERPBase=ResidualSphericalLERPBase,
            ResidualAdaptiveSphericalLERP=ResidualAdaptiveSphericalLERP,
            ResidualSphericalSLERP=ResidualSphericalSLERP,
            ResidualAdaptiveSphericalSLERP=ResidualAdaptiveSphericalSLERP)

        ResidualModule = residual_module_dict[residual_module]
        residual_module_kwargs = residual_module_kwargs or {}
        if residual_module in ['ResidualAdaptiveSphericalLERP', 'ResidualAdaptiveSphericalSLERP']:
            residual_module_kwargs['parametrize'] = not manual_norm_weights

        for (
            alpha_attn_init_,
            alpha_attn_scale_,
            alpha_ff_init_,
            alpha_ff_scale_,
            s_qk_init_,
            s_qk_scale_,
            s_ff_hidden_init_,
            s_ff_hidden_scale_,
            s_ff_gate_init_,
            s_ff_gate_scale_
        ) in zip(*scale_hparams):

            attn = Attention(
                dim,
                dim_head = dim_head,
                heads = heads,
                causal = causal,
                norm_qk = attn_norm_qk,
                manual_norm_weights = manual_norm_weights,
                s_qk_init = s_qk_init_,
                s_qk_scale = s_qk_scale_,
                flash_kwargs = attn_flash_kwargs,
                norm_eps = norm_eps,
                num_hyperspheres = num_hyperspheres
            )

            ff = FeedForward(
                dim,
                expand_factor = ff_expand_factor,
                manual_norm_weights = manual_norm_weights,
                s_hidden_init = s_ff_hidden_init_,
                s_hidden_scale = s_ff_hidden_scale_,
                s_gate_init = s_ff_gate_init_,
                s_gate_scale = s_ff_gate_scale_,
                norm_eps = norm_eps,
                num_hyperspheres = num_hyperspheres
            )

            if residual_module == 'ResidualSphericalLERPBase':
                residual_module_kwargs = dict(
                    init=default(alpha_attn_init_, alpha_init),
                    scale=default(alpha_attn_scale_, dim ** -0.5)
                )
            attn_with_residual = ResidualModule(
                attn,
                dim,
                **residual_module_kwargs
            )

            if residual_module == 'ResidualSphericalLERPBase':
                residual_module_kwargs = dict(
                    init=default(alpha_ff_init_, alpha_init),
                    scale=default(alpha_ff_scale_, dim ** -0.5)
                )

            ff_with_residual = ResidualModule(
                ff,
                dim,
                **residual_module_kwargs
            )

            self.layers.append(ModuleList([attn_with_residual, ff_with_residual]))

        self.to_logits = NormLinear_(dim, vocab_size) if not tied_embedding else None

        self.logit_scale = Scale(vocab_size, s_logit_init, default(s_logit_scale, dim ** -0.5))

        self.ignore_index = ce_ignore_index

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            module.norm_weights_()

    def register_step_post_hook(self, optimizer):
        """
        This function registers a hook to an optimizer that will normalize the weights of the model after each step.

        This should be used if manual_norm_weights = True or equivalently if paramtrize = False in NormLinear.

        This represents two possible ways to maintain normalized weights in the model:
        1) using the parametrize argument in NormLinear, to use a unit-norm parameterization, and differentiate through the parameterization
        2) using a post-step hook to normalize the weights after each optimization step
        """

        assert hasattr(optimizer, 'register_step_post_hook')

        def hook(*_):
            self.norm_weights_()

        return optimizer.register_step_post_hook(hook)

    def forward(
        self,
        ids,
        mask = None,
        return_loss = False
    ):
        token_embed, rotary_embed = self.token_embed.weight, self.rotary_embed

        if return_loss:
            assert self.causal
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = token_embed[ids]

        first_values = None

        for attn, ff in self.layers:
            tokens, values = attn(tokens, mask = mask, rotary_embed = rotary_embed, return_values = True, value_residual = first_values if self.add_value_residual else None)

            first_values = default(first_values, values)

            tokens = ff(tokens)

        if exists(self.to_logits):
            logits = self.to_logits(tokens)
        else:
            # tied embeddings
            logits = einsum(tokens, token_embed, 'b n d, c d -> b n c')

        logits = logits * self.logit_scale()

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss