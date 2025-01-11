# Source: https://github.com/lucidrains/nGPT-pytorch/blob/5208aada3330e366c74ce21f4701f8d2b6aa5761/nGPT_pytorch/nGPT.py

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
import math

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from .nGPT import Attention, FeedForward
from .rotary_embedding_torch import RotaryEmbedding
from .residual_stream import ResidualSphericalLERPBase, ResidualAdaptiveSphericalLERP, ResidualSphericalSLERP, ResidualAdaptiveSphericalSLERP
from .norm_utils import NormLinear, Scale, L2Norm, l2norm
from utils.utils import default, exists, cast_tuple

# constants

class RecurrentnGPT(Module):
    def __init__(
        self,
        *,
        vocab_size,
        dim,
        depth,
        n_iters,
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
        norm_eps = 0., # greater than 0 allows the norm to be around (1. - norm_eps) to (1. + norm_eps)
        gpt_special_init=False # temporary flag for initialization (FIXME: in principle, should be unncessary)
    ):
        super().__init__()
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights, norm_eps = norm_eps, groups = num_hyperspheres)
        self.l2norm = partial(l2norm, norm_eps = norm_eps, groups = num_hyperspheres)

        self.dim = dim
        self.n_iters = n_iters
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

        # TODO: special initialization??
        # initialize weights
        if gpt_special_init:
            self.apply(self._init_weights)

            for pn, p in self.named_parameters():
                if 'to_out' in pn and pn.endswith('.weight'):
                    torch.nn.init.normal_(p,
                        mean=0.0, std=0.02 / math.sqrt(2 * depth))

        # norm weights
        self.norm_weights_()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        for _ in range(self.n_iters):
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