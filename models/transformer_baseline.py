import torch
import torch.nn as nn

from einops import rearrange

import math

from functools import partial
from typing import Any, Optional, Tuple

# region Language Models

class TransformerLM(torch.nn.Module):
    """
    Transformer Language Model.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary containing the following keys:
        - d_model (int): Dimension of the model.
        - n_heads (int): Number of attention heads.
        - n_layers (int): Number of Transformer layers.
        - dff (int): Dimension of the feed-forward layer.
        - mlp_activation (str): Activation function for the feed-forward layer.
        - norm_config (dict): Configuration for normalization layers.
        - vocab_size (int): Size of the vocabulary.
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.

    Methods
    -------
    get_pos_enc_model(attn=True):
        Returns the positional encoding model based on the configuration.
    forward(x):
        Forward pass of the model.
    """

    def __init__(self, model_config):

        super(TransformerLM, self).__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.norm_config = getattr(model_config, 'norm_config', None)
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})
        self.tied_embedding = getattr(model_config, 'tied_embedding', False) # weight tying
        self.gpt_special_init = getattr(model_config, 'gpt_special_init', False) # special initialization for GPT-2

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = self.get_pos_enc_model() if self.pos_enc_type in ['sinusoidal', 'learned'] else None

        self.token_embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)

        self.blocks = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=self.get_pos_enc_model(attn=True), # positional encoding model for attention (e.g., RoPE, T5, etc.)
            attn_kwargs=self.attn_kwargs, causal=True)
            for _ in range(model_config.n_layers)])

        # if using pre-norm, apply layernorm before final linear layer
        if model_config.norm_config.norm_method == 'pre-norm':
            self.prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        else:
            self.prelogits_norm = torch.nn.Identity()

        self.embed_to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        # weight tying between token_embedder and embed_to_token_logits
        if self.tied_embedding:
            self.embed_to_token_logits.weight = self.token_embedder.weight

        # initialize weights
        if self.gpt_special_init:
            self.apply(self._init_weights)

            # per-GPT2 paper, scale intialization of output projection and last layer of mlp
            # apply special n_layer-scaled initialization to layers that add to the residual stream
            # (output projection of attention and last layer of mlp)
            # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
            # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
            mlp_special_init_layer = 'linear3' if self.model_config.mlp_activation == 'swiglu' else 'linear2'
            for pn, p in self.named_parameters():
                if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                    torch.nn.init.normal_(p,
                        mean=0.0, std=0.02 / math.sqrt(2 * self.model_config.n_layers))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_enc_model(self, attn=True):
        if self.pos_enc_type == 'sinusoidal' and not attn:
            return ScaledSinusoidalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'learned' and not attn:
            return AbsolutePositionalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'alibi':
            return AlibiPositionalBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify slopes in pos_enc_kwargs
        elif self.pos_enc_type == 't5':
            return T5RelativePositionBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
        elif self.pos_enc_type == 'rotary':
            return RotaryPositionalEmbeddings(dim=self.d_head, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'none' or ((self.pos_enc_type in ['sinusoidal', 'learned']) and attn):
            return None
        else:
            raise ValueError(f"pos_enc_type {self.pos_enc_type} not recognized")

    def forward(self, x):

        # embed tokens
        x = self.token_embedder(x)

        # if positional encoding model is additive-embedding-based, add it to the input
        if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
            x += self.pos_enc_model(x)

        # apply the Transformer layers
        for encoder in self.blocks:
            x = encoder(x)

        # apply pre-logits normalization (if using pre-norm)
        x = self.prelogits_norm(x)

        # project to logits
        logits = self.embed_to_token_logits(x)

        return logits

class RecurrentTransformerLM(torch.nn.Module):
    """
    Recurrent Transformer Language Model.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary containing the following keys:
        - d_model (int): Dimension of the model.
        - n_heads (int): Number of attention heads.
        - n_layers (int): Number of Transformer layers.
        - n_iters (int): Number of iterations to run the model (by default).
        - dff (int): Dimension of the feed-forward layer.
        - bias (bool): Whether to use bias in the linear layers.
        - mlp_activation (str): Activation function for the feed-forward layer.
        - norm_config (dict): Configuration for normalization layers.
        - vocab_size (int): Size of the vocabulary.
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.

    Methods
    -------
    get_pos_enc_model(attn=True):
        Returns the positional encoding model based on the configuration.
    forward(x):
        Forward pass of the model.
    """

    def __init__(self, model_config):

        super().__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.bias = model_config.bias
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.norm_config = getattr(model_config, 'norm_config', None)
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})
        self.tied_embedding = getattr(model_config, 'tied_embedding', False) # weight tying
        self.n_iters = model_config.n_iters # number of iterations to run the model
        self.gpt_special_init = getattr(model_config, 'gpt_special_init', False) # special initialization for GPT-2

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = self.get_pos_enc_model() if self.pos_enc_type in ['sinusoidal', 'learned'] else None

        self.token_embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)

        self.blocks = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=self.get_pos_enc_model(attn=True), # positional encoding model for attention (e.g., RoPE, T5, etc.)
            attn_kwargs=self.attn_kwargs, bias=self.bias, causal=True)
            for _ in range(model_config.n_layers)])

        # if using pre-norm, apply layernorm before final linear layer
        if model_config.norm_config.norm_method == 'pre-norm':
            self.prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        else:
            self.prelogits_norm = torch.nn.Identity()

        self.embed_to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size, bias=self.bias)

        # weight tying between token_embedder and embed_to_token_logits
        if self.tied_embedding:
            self.embed_to_token_logits.weight = self.token_embedder.weight

        # initialize weights
        if self.gpt_special_init:
            self.apply(self._init_weights)

            # per-GPT2 paper, scale intialization of output projection and last layer of mlp
            # apply special n_layer-scaled initialization to layers that add to the residual stream
            # (output projection of attention and last layer of mlp)
            # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
            # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
            mlp_special_init_layer = 'linear3' if self.model_config.mlp_activation == 'swiglu' else 'linear2'
            for pn, p in self.named_parameters():
                if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                    torch.nn.init.normal_(p,
                        mean=0.0, std=0.02 / math.sqrt(2 * self.model_config.n_layers))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_enc_model(self, attn=True):
        if self.pos_enc_type == 'sinusoidal' and not attn:
            return ScaledSinusoidalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'learned' and not attn:
            return AbsolutePositionalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'alibi':
            return AlibiPositionalBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify slopes in pos_enc_kwargs
        elif self.pos_enc_type == 't5':
            return T5RelativePositionBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
        elif self.pos_enc_type == 'rotary':
            return RotaryPositionalEmbeddings(dim=self.d_head, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'none' or ((self.pos_enc_type in ['sinusoidal', 'learned']) and attn):
            return None
        else:
            raise ValueError(f"pos_enc_type {self.pos_enc_type} not recognized")

    def forward(self, x, n_iters=None):

        n_iters = self.n_iters if n_iters is None else n_iters

        # embed tokens
        x = self.token_embedder(x)

        # if positional encoding model is additive-embedding-based, add it to the input
        if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
            x += self.pos_enc_model(x)

        # apply the Transformer layers
        for iter in range(n_iters):
            for encoder in self.blocks:
                x = encoder(x)

        # apply pre-logits normalization (if using pre-norm)
        x = self.prelogits_norm(x)

        # project to logits
        logits = self.embed_to_token_logits(x)

        return logits


# endregion

# region Transformer Blocks

class EncoderBlock(nn.Module):

    def __init__(self,
            d_model: int,
            n_heads: int,
            pos_enc_model = None,
            dff: int = None,
            activation: str = 'relu',
            norm_config: dict = None,
            dropout_rate: float = 0.0,
            bias: bool = True,
            causal: bool = False,
            attn_kwargs: dict = None,
            ):
        """
        A Transformer Encoder Block.

        Consists of Self-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        pos_enc_model : nn.Module, optional
            positional encoding model to use. Default is None.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feed-forward block.
        norm_config: dict, optional
            norm_type: specifies type of normalization to use ('layernorm' or 'rmsnorm'). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention (e.g., 'pre-norm', 'post-norm', etc.). Default is 'pre-norm'.
        dropout_rate : float
            dropout rate.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_config = dict(norm_method='pre-norm', norm_type='layernorm') | (norm_config or {})
        self.bias = bias
        self.attn_kwargs = {'n_kv_heads': None, 'add_bias_kv': False}
        if attn_kwargs is not None:
            self.attn_kwargs.update(attn_kwargs)
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.residual_stream_block_attn = ResidualStreamBlock(self.d_model, norm_config=self.norm_config)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)

        self.residual_stream_block_ff = ResidualStreamBlock(self.d_model, norm_config=self.norm_config)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)


    def forward(self, x, need_weights=False, need_intermediate=False):

        intermediate_outputs = dict() if need_intermediate else None

        # self-attention + residual + norm
        x = self.residual_stream_block_attn(x, partial(self._compute_self_attn, intermediate_outputs=intermediate_outputs), need_weights=need_weights)

        # feed-forward + residual + norm
        x = self.residual_stream_block_ff(x, self._apply_ff_block)

        if need_intermediate:
            return x, intermediate_outputs

        return x

    def _compute_self_attn(self, x, need_weights=False, intermediate_outputs=None):

        # if intermediate_outputs is not None, store attention scores
        if intermediate_outputs is not None:
            need_weights = True

        x, attn_scores = self.self_attn(query=x, key=x, value=x, is_causal=self.causal,
            need_weights=need_weights)

        if intermediate_outputs is not None:
            intermediate_outputs['self_attn_scores'] = attn_scores

        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_heads_cross: int,
            pos_enc_model_sa = None,
            pos_enc_model_ca = None,
            dff: int = None,
            activation: str = 'relu',
            norm_config: dict = None,
            dropout_rate: float = 0.,
            bias: bool = True,
            causal: bool = False,
            attn_kwargs: dict = None,
            ):
        """
        A Transformer Decoder Block.

        Consists of Self-attention, Cross-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        n_heads_cross : int
            number of cross-attention heads.
        pos_enc_model_sa : nn.Module, optional
            positional encoding model to use. Default is None.
        pos_enc_model_ca : nn.Module, optional
            positional encoding model to use. Default is None.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feed-forward block.
        norm_config: dict, optional
            norm_type: specifies type of normalization to use ('layernorm' or 'rmsnorm'). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention ('pre-norm' or 'post-norm'). Default is 'pre-norm'.
        dropout_rate : float
            dropout rate.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        attn_kwargs : dict, optional
            keyword arguments for Attention, by default None
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_heads_cross = n_heads_cross
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_config = dict(norm_method='pre-norm', norm_type='layernorm') | (norm_config or {})
        self.bias = bias
        self.causal = causal
        self.attn_kwargs = {'n_kv_heads': None, 'add_bias_kv': False}
        if attn_kwargs is not None:
            self.attn_kwargs.update(attn_kwargs)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.residual_stream_block_selfattn = ResidualStreamBlock(self.d_model, norm_config=self.norm_config)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model_sa,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)

        self.residual_stream_block_crossattn = ResidualStreamBlock(self.d_model, norm_config=self.norm_config)
        self.cross_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads_cross, pos_enc_model=pos_enc_model_ca,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)

        self.residual_stream_block_ff = ResidualStreamBlock(self.d_model, norm_config=self.norm_config)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, context, need_intermediate=False):

        intermediate_outputs = dict() if need_intermediate else None

        # self-attention + residual + norm
        x = self.residual_stream_block_selfattn(x, partial(self._compute_self_attn, intermediate_outputs=intermediate_outputs))

        # cross-attention + residual + norm
        x = self.residual_stream_block_crossattn(x, partial(self._compute_cross_attn, intermediate_outputs=intermediate_outputs), context=context)

        # feed-forward + residual + norm
        x = self.residual_stream_block_ff(x, self._apply_ff_block)

        if need_intermediate:
            return x, intermediate_outputs

        return x

    def _compute_self_attn(self, x, need_weights=False, intermediate_outputs=None):
        # if intermediate_outputs is not None, store attention scores
        if intermediate_outputs is not None:
            need_weights = True

        x, attn_scores = self.self_attn(query=x, key=x, value=x,
            is_causal=self.causal, need_weights=need_weights)

        if intermediate_outputs is not None:
            intermediate_outputs['self_attn_scores'] = attn_scores

        x = self.dropout(x)
        return x

    def _compute_cross_attn(self, x, context, need_weights=False, intermediate_outputs=None):
        # if intermediate_outputs is not None, store attention scores
        if intermediate_outputs is not None:
            need_weights = True

        x, attn_scores = self.cross_attn(query=x, key=context, value=context,
            is_causal=False, need_weights=need_weights)

        if intermediate_outputs is not None:
            intermediate_outputs['cross_attn_scores'] = attn_scores

        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x


class FeedForwardBlock(nn.Module):

    def __init__(self,
            embed_dim: int,
            dff: int = None,
            activation: str = 'relu',
            use_bias: bool = False):
        """
        Feed-forward block.

        A 2-layer neural network with activation function in between.

        Parameters
        ----------
        embed_dim : int
            embedding dimension of input.
        dff : int, optional
            size of intermediate layer. if None, 4 * embed_dim.
        activation : str, optional
            name of activation function, by default 'relu'
        use_bias : bool, optional
            whether to use bias in linear layers, by default False
        """

        super().__init__()
        self.embed_dim = embed_dim

        # set dff according to activation function if not given
        if dff is None and activation == 'swiglu':
            self.dff = int(2/3 * 4 * embed_dim)
        elif dff is None:
            self.dff = 4 * embed_dim
        else:
            self.dff = dff

        self.use_bias = use_bias
        self.activation = activation
        if self.activation != 'swiglu':
            self.activation_ = get_activation_function(activation)

        self.linear1 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)
        self.linear2 = nn.Linear(self.dff, self.embed_dim, bias=self.use_bias)
        if self.activation == 'swiglu':
            self.linear3 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)

    def forward(self, x):
        if self.activation == 'swiglu':
            return self.linear2(nn.functional.silu(self.linear1(x)) * self.linear3(x))
        else:
            x = self.linear1(x)
            x = self.activation_(x)
            x = self.linear2(x)
            return x

def get_activation_function(name):
    """gets activation function by its name."""

    activation_dict = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),

        # Gaussian Error Linear Unit: GELU(x) = x * GaussianCDF(x)
        'gelu': nn.GELU(approximate='tanh'),

        # Sigmoid Linear Unit: silu(x) = x * sigmoid(x)
        'silu': nn.SiLU(),

        # Softmax of Linear Units: SoLU(x) = x * softmax(x)
        # (https://transformer-circuits.pub/2022/solu/index.html)
        'solu': lambda x: x * torch.nn.functional.softmax(x, dim=-1),

        # LayerNormed Softmax of Linear Units: LNSoLU(x) = LN(x * softmax(x))
        # Note: here, I am using layernorm functional with no learnable parameters;
        # Note: interpretable activations would be pre-norm post-softmax (i.e., the SoLU part)
        # NOTE: this is equivalent to the original SoLU(x) = LN(x * exp(x)) in terms of final model performance due to scale-invariance of LayerNorm
        'lnsolu': lambda x: torch.nn.functional.layer_norm(x * torch.nn.functional.softmax(x, dim=-1)),

        'softmax': nn.Softmax(dim=-1),
        'identity': nn.Identity(),
        # add more if needed
    }
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise ValueError(f'Activation function {name} not found in {activation_dict.keys()}')

# endregion

# region positional encoding methods

import sys
import os
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, ModuleDict

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

import os, sys; sys.path.insert(0, os.path.abspath('..')) # add project root dir to path
from utils.utils import exists, default, l2norm, pad_at_dim, Sequential

# region
# code in this region is based on https://github.com/lucidrains/x-transformers/blob/144d9ba84955139347e798ab025457b2d7adc314/x_transformers/x_transformers.py (November 8, 2024)


class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert (
            seq_len <= self.max_seq_len
        ), f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (
            dim % 2 == 0
        ), "dimension of the model must be divisible by 2 for sinusoidal positional encoding"
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


# This is adapted from Mesh Tensorflow: 
# https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
# perhaps via Huggingface Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L344

class T5RelativePositionBias(Module):
    def __init__(self,  heads, scale=1, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal, num_buckets, max_distance):
        """
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.
        If causal=True, then positive relative positions are
        invalid.

        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            causal: a boolean - whether the attention is causal
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, qlen, klen):
        device = self.device
        q_pos = torch.arange(klen - qlen, klen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)

        rel_pos = einx.subtract("j, i -> i j", k_pos, q_pos)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """

        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """

    def __init__(
        self,
        dim,
        heads,
        max_pos,
        soft_onehot=False,
        talking_heads=False,
        soft_onehot_temp=5e-2,
    ):
        super().__init__()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None
        )
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer("positions", torch.arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(
                causal_mask, -torch.finfo(attn_logits.dtype).max
            )

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.max_pos - 1)

        logits_int = einsum("b h n d, p d -> b h n p", query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract("i, j -> i j", pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim=-1)
            cope_pos_emb = einsum(
                "b h i j p, b h i p -> b h i j", soft_onehot_pos, logits_int
            )
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb


class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert (
            depth >= 1
        ), "depth for dynamic position bias MLP must be greater or equal to 1"
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(
            Sequential(nn.Linear(1, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU())
        )

        for _ in range(depth - 1):
            self.mlp.append(
                Sequential(
                    nn.Linear(dim, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU()
                )
            )

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, qlen, klen):
        assert qlen == klen
        n, device = klen, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = einx.subtract("i, j -> i j", seq_arange, context_arange)
        indices += n - 1

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(
                pos.abs() + 1
            )  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


class AlibiPositionalBias(Module):
    def __init__(
        self, heads, total_heads=None, slopes: list[int] | None = None, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, "h -> h 1 1")

        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward_custom_pos(self, pos_i: Tensor, pos_j: Tensor | None = None):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract("... j, ... i -> ... i j", pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, "b i j -> b 1 i j")

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        return bias

    def forward(self, qlen, klen):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= klen and self.bias.shape[-2] >= qlen:
            return self.bias[..., -qlen:, -klen:]

        seq_arange = torch.arange(klen - qlen, klen, device=device)
        context_arange = torch.arange(klen, device=device)
        bias = -einx.subtract("j, i -> 1 i j", context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class DataDependentAlibi(Module):
    """https://openreview.net/forum?id=q2Lnyegkr8"""

    def __init__(
        self,
        dim,
        heads,
        causal=True,
        bias_init=5.0,
        post_log_scale=1.0,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n h -> b h n"), nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim=-1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim=1)

        forget_gates = einx.subtract(
            "b h i, b h j -> b h i j", forget_gates, forget_gates
        )

        if bidirectional:
            forget_gates_reversed = einx.subtract(
                "b h j, b h i -> b h i j", forget_gates_reversed, forget_gates_reversed
            )
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates


class PerRowDataDependentAlibi(Module):
    """same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension"""

    def __init__(self, dim, heads, causal=True, dim_head=8, post_log_scale=1.0):
        super().__init__()
        assert causal, "bidirectional not supported yet"

        self.scale = dim_head**-0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias=False)

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n (qk h d) -> qk b h n d", qk=2, d=dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum("... i d, ... j d -> ... i j", q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype=torch.bool, device=x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.0)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims=(-1,))
        forget_gates = forget_gates.cumsum(dim=-1)
        forget_gates = forget_gates.flip(dims=(-1,))

        return forget_gates


# endregion


# this implementation is from the torchtune package, and is based on the official llama source code
# https://github.com/pytorch/torchtune/blob/bce70917c3d0d1f7693c9ae8b59cd72ee55b659d/torchtune/modules/position_embeddings.py (11/11/2024)
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

def get_pos_enc_model(pos_enc_type, pos_enc_kwargs, attn=False):
    # attn is convenience flag which returns None if pos_enc_type is set to 'sinusoidal' or 'learned'

    if pos_enc_type == 'sinusoidal' and not attn:
        return ScaledSinusoidalEmbedding(**pos_enc_kwargs) # required: dim=d_model
    elif pos_enc_type == 'learned' and not attn:
        return AbsolutePositionalEmbedding(**pos_enc_kwargs) # required: dim=d_model
    elif pos_enc_type == 'alibi':
        return AlibiPositionalBias(**pos_enc_kwargs) # required: n_heads; can specify slopes in pos_enc_kwargs
    elif pos_enc_type == 't5':
        return T5RelativePositionBias(**pos_enc_kwargs) # required: n_heads; can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
    elif pos_enc_type == 'rotary':
        return RotaryPositionalEmbeddings(**pos_enc_kwargs) # required: dim=dim_head
    elif pos_enc_type == 'none' or ((pos_enc_type in ['sinusoidal', 'learned']) and attn):
        return None
    else:
        raise ValueError(f"pos_enc_type {pos_enc_type} not recognized")

# endregion

# region attention

class Attention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            pos_enc_model: nn.Module = None,
            key_dim: int = None,
            n_kv_heads: int = None,
            dropout: float = 0.0,
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            symmetric_attn: bool = False,
            attn_score_fn: str = 'softmax',
            attn_score_fn_params: dict = None,
            ):
        """
        An implementation of Attention with some added customization.

        Allows multi-query attention/grouped query attention, rotary positional embeddings,
        and custom relation activation functions.

        Parameters
        ----------
        d_model : int
            model dimension
        n_heads : int
            number of heads (query heads if n_kv_heads is set)
        pos_enc_model : nn.Module, optional
            positional encoding model, e.g., RoPE, T5RelativePositionalBias, etc. (default is None)
        dropout : float
            dropout rate
        n_kv_heads : int, optional
            number of key/value heads. used to implement multi-query attention or grouped query attention.
            n_kv_heads=1 corresponds to MQA, n_kv_heads > 1 corresponsd to grouped query attention.
            n_kv_heads=n_heads is standard MHA. uses MHA when None. By default None
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        symmetric_attn : bool, optional
            whether to weight-tie the query and key projections, making a symmetric attention criterion. By default False
        attn_score_fn : str, optional
            activation function for attention scores. One of 'softmax', 'hard', 'topk-softmax', 'sigmoid', or 'linear' (default is 'softmax').
        attn_score_fn_params : dict, optional
            additional parameters for the attention score function, e.g., whether to use straight-through estimator for sparse softmax variants, etc. (default is None)
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # number of heads (for query)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.symmetric_attn = symmetric_attn

        self.pos_enc_model = pos_enc_model
        self.pos_enc_model_type = get_pos_enc_model_type(pos_enc_model)

        self.key_dim = key_dim if key_dim is not None else self.d_model // self.n_heads # key dimension
        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.n_heads # dim of projections
        assert self.n_heads % self.n_kv_heads == 0 # make sure n_kv_heads fits into n_heads (i.e., can be grouped)
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.key_dim) # for scaled dot product attention

        self.wq = nn.Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)
        if symmetric_attn:
            self.wk = self.wq
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # activation function for attention scores (e.g., softmax, hard, topk-softmax, sigmoid, linear)
        self.attn_score_fn = attn_score_fn
        self.attn_score_fn_params = attn_score_fn_params or {}
        self.attn_score_fn_ = get_attention_function(self.attn_score_fn, self.attn_score_fn_params)

        # check whether configuration (namely positional encoding model and attention score function) supports flash attention
        self.support_flash = self.is_flash_supported()

    def is_flash_supported(self):
        pos_enc_support = get_pos_enc_support(self.pos_enc_model)
        attn_func_support = self.attn_score_fn == 'softmax'
        return pos_enc_support['flash'] and attn_func_support

    def create_attn_score_mod(self, bias=None):
        if bias is not None:
            def score_mod(score, b, h, q_idx, kv_idx):
                score_bias = bias[h, q_idx, kv_idx]
                return score + score_bias
        else:
            score_mod = None

        return score_mod

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_func: callable = None,
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and mask_func
        need_weights: bool = False
    ):
        """
        compute attention with given query, key, value.

        if freqs_cos and freqs_sin are given, apply rotary positional embeddings.
        if attn_mask is given, apply attention mask.
        if is_causal is True, apply causal mask (attn_mask must be None).

        Parameters
        ----------
        query : torch.Tensor
            query sequence of shape [bsz, len_in, d_model]
        key : torch.Tensor
            key sequence of shape [bsz, len_ctx, d_model]
        value : torch.Tensor
            value sequence of shape [bsz, len_ctx, d_model]
        mask_func : callable, optional
            mask_mod function. This is a callable that defines the masking pattern for the attention mechanism. It takes four arguments: b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index). It should return a boolean tensor indicating which attention connections are allowed (True) or masked out (False).
        is_causal : bool, optional
            whether to apply a causal mask. If True, mask_func must be None. Only applies for self-attention.
            By default False
        need_weights : bool, optional
            whether to return the attention scores. If True, return value will be tuple (output, attn_scores).
            If True, will compute attention manually rather than using flash attention. By default False

        Returns
        -------
        torch.Tensor
            result of attention
        """

        bsz, qseqlen, _ = query.shape
        bsz, kseqlen, _ = key.shape
        bsz, vseqlen, _ = value.shape

        assert kseqlen == vseqlen, "key and value sequences must have the same length"

        assert not (mask_func is not None and is_causal), "only one of attn_mask and is_causal should be set"
        # compute causal mask if is_causal and no maks given
        if is_causal and mask_func is None:
            assert qseqlen == kseqlen, "query and key sequences must have the same length for causal mask"
            attn_mask = compute_causal_mask(qseqlen, device=query.device)
        elif not is_causal and mask_func is not None:
            # TODO: avoid flex_attention dependency now that I've removed it
            attn_mask = torch.nn.attention.flex_attention.create_mask(
                mask_func, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kseqlen, device=query.device)
        else:
            attn_mask = None

        # apply query/key/value projections and reshape to split into different heads
        xq, xk, xv = self.wq(query), self.wk(key), self.wv(value)
        xq = xq.view(bsz, qseqlen, self.n_heads, self.key_dim) # shape (bs, seqlen, n_heads, key_dim)
        xk = xk.view(bsz, kseqlen, self.n_kv_heads, self.key_dim) # shape (bs, seqlen, n_kv_heads, key_dim)
        xv = xv.view(bsz, vseqlen, self.n_kv_heads, self.head_dim) # shape (bs, seqlen, n_kv_heads, head_dim)

        # apply RoPE to queries and keys (if positional encoding model is RoPE)
        if self.pos_enc_model_type == 'rotary':
            # recall that RotaryPositionalEmbeddings expects an input of shape (bs, seqlen, n_heads, key_dim)
            xq = self.pos_enc_model(xq)
            xk = self.pos_enc_model(xk)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)  # (bs, seqlen, n_heads, key_dim)
            xv = repeat_kv(xv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, key_dim)
        xk = xk.transpose(1, 2)  # (bs, n_heads, seqlen, key_dim)
        xv = xv.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)

        # determine whether to use flash attention or manual attention
        # use flash attention if no need for weights and positional encoding method supports it
        use_flash_attn = (not need_weights) and self.support_flash and (not mask_func)

        # can use F.scaled_dot_product_attention's implementation of flash attention
        if use_flash_attn:

            # fixed bias-based positional encoding method (e.g., AlibiPositionalBias)
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
                if attn_mask is not None:
                    mask_bias = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                    scores_bias = scores_bias + mask_bias

                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=scores_bias, dropout_p=self.dropout if self.training else 0.0, scale=self.attn_scale)

            # pos enc already applied to xq and/or xk (e.g., RoPE or NoPE)
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.attn_scale)
            scores = None

        # manual implementation (which explicitly computes attention scores)
        else:
            # compute dot product attention scores (pre-softmax)
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

            # apply bias-based positional encoding (if applicable)
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
                scores = scores + scores_bias

            if attn_mask is not None and self.attn_score_fn in ['softmax', 'topk-softmax', 'hard']:
                attn_mask_ = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                scores = scores + attn_mask_

            # apply softmax (or other) activation to inner products
            scores = self.attn_score_fn_(scores)

            if attn_mask is not None and self.attn_score_fn not in ['softmax', 'topk-softmax', 'hard']:
                scores = scores.masked_fill(attn_mask.logical_not(), 0)

            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, qseqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores

# these position encoding models have an interface of the form (qseqlen: int, kseqlen: int) -> Tensor[n_heads, qseqlen, kseqlen]
def get_pos_enc_model_type(pos_enc_model):
    # this groups positional encoding models into categories based on their interface
    if any(isinstance(pos_enc_model, model) for model in [AlibiPositionalBias, T5RelativePositionBias]):
        return 'score_bias'
    elif isinstance(pos_enc_model, RotaryPositionalEmbeddings):
        return 'rotary'
    elif pos_enc_model is None:
        return None
    elif any(isinstance(pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
        return None
    else:
        raise ValueError(f"unknown positional encoding model: {pos_enc_model}")

def get_pos_enc_support(pos_enc_model):
    flash_support = [RotaryPositionalEmbeddings, AlibiPositionalBias]
    # NOTE: T5RelativePositionBias does not support flash attention because flash attention requires a fixed bias (cannot backprop)

    support_dict = dict(
        flash=any(isinstance(pos_enc_model, model) for model in flash_support), # positional encoding methods that support flash attention
        manual=True # all support manual
        )
    if pos_enc_model is None:
        support_dict['flash'] = True
    return support_dict

# endregion

# region attention_utils

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# NOTE: be careful. pytorch API is inconsistent about whether True means attend or not attend. 
# this works with the Attention module implemented above, but will only be compatible with some but not all pytorch implementations
# e.g., works with nn.functional.scaled_dot_product_attention but not nn.MultiHeadAttention
def compute_diag_mask(size, device=None):
    """computes an attention mask with False on the diagonal and True elsewhere"""

    diag_mask = torch.eye(size, device=device).logical_not()
    # diag_mask = diag_mask.masked_fill(diag_mask == 1, float('-inf'))
    return diag_mask

def compute_causal_mask(size, device=None):
    """computes an attention mask with True at (i,j) if i <= j"""
    causal_mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return causal_mask

def topk_softmax(logits: torch.Tensor, k: int, straight_through: bool = False) -> torch.Tensor:
    """
    Apply top-k softmax to the logits.

    Parameters
    ----------
    logits : torch.Tensor
        [batch_size, n_heads, seq_len, seq_len] tensor of logits.
    k : int
        The number of top elements to consider.
    straight_through : bool, optional
        Whether to use the straight-through estimator (default is False).

    Returns
    -------
    torch.Tensor
        topk-softmax attention scores.
    """

    orig_logits = logits

    mask_value = -torch.finfo(logits.dtype).max
    top_values, _ = logits.topk(k, dim = -1)
    sparse_topk_mask = (logits >= top_values[..., -1:]) & (logits > mask_value)
    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    topk_attn = logits.softmax(dim = -1)

    if straight_through:
        # straight-through estimator: value returned is topk_attn, but gradient is soft_attn

        soft_attn = orig_logits.softmax(dim = -1)
        return topk_attn.detach() + soft_attn - soft_attn.detach()
    else:
        return topk_attn

def get_attention_function(activation: str, kwargs: dict) -> Any:
    """
    Get the attention function based on the activation.

    Parameters
    ----------
    activation : str
        The activation function.
    kwargs : dict
        The keyword arguments for the activation function (if applicable).

    Returns
    -------
    Any
        The attention function.
    """

    if activation == "softmax":
        return partial(torch.nn.functional.softmax, dim=-1)
    elif activation == "topk-softmax":
        return partial(topk_softmax, **kwargs)
    elif activation == "hard":
        return partial(topk_softmax, k=1, **kwargs)
    elif activation == "sigmoid":
        return torch.nn.functional.sigmoid
    elif activation == "relu":
        return torch.nn.functional.relu
    elif activation == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Activation function {activation} not valid.")

# endregion

# region ResidualStream

VALID_NORM_METHODS = ['pre-norm', 'post-norm', 'pre+post-norm', 'hypersphere-interpolation', 'hypersphere-spherical-interpolation', 'adaptive-hypersphere-interpolation', 'none']
class ResidualStreamBlock(nn.Module):
    def __init__(self, dim, norm_config=None):
        """This Module applies a residual connection to the input of a model_func, with optional normalization before and/or after the model_func.

        E.g., implements y = x + model_func(norm(x)), in case of pre-norm.

        Parameters
        ----------
        dim : int
            Dimension of the input tenso and output tensors (e.g, d_model)
        norm_config : dict, optional
            norm_type: specifies type of normalization to use (see create_norm for options). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention ('none, 'pre-norm', 'post-norm', or 'pre+post-norm'). Default is 'pre-norm'.
        """
        super().__init__()

        self.dim = dim
        self.norm_config = norm_config or {}
        self.norm_method = self.norm_config.get('norm_method', 'pre-norm') # 'pre-norm' or 'post-norm' or 'none'
        assert self.norm_method in VALID_NORM_METHODS, f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}'

        self.norm_type = self.norm_config.get('norm_type', 'layernorm')

        if self.norm_method in ['pre-norm', 'post-norm']:
            self.norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None

        elif self.norm_method == 'pre+post-norm':
            self.pre_norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None
            self.post_norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None

        elif self.norm_method == 'hypersphere-interpolation':
            # use only valid kwargs in norm_config
            self.res_stream = HypersphereLERP(dim, lerp_weight_constraint=self.norm_config.get('lerp_weight_constraint', 'none'))
            # lerp_scale = self.norm_config.get('lerp_scale', self.dim ** 0.5)
            # lerp_init = self.norm_config.get('lerp_init', 1.0) # NOTE: can be set 1 / n_layers
            # self.forward_lerp_weight_scale = lerp_init / lerp_scale
            # self.lerp_weight = nn.Parameter(torch.ones(self.dim) * lerp_scale, requires_grad=True)

            # note: norm_type is not used here, we always normalize to unit-norm hypersphere
        elif self.norm_method == 'hypersphere-spherical-interpolation':
            self.res_stream = HypersphereSLERP(dim, single_weight=self.norm_config.get('single_weight', True))

        elif self.norm_method == 'adaptive-hypersphere-interpolation':
            self.res_stream = AdaptiveHypersphereSLERP(dim, single_weight=self.norm_config.get('single_weight', True))

        elif self.norm_method == 'none':
            pass
        else:
            raise ValueError(f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}')

        self.dim = dim


    def forward(self, x, model_func, **model_kwargs):

        if self.norm_method == 'none':
            y = model_func(x, **model_kwargs)
            x = x + y

        elif self.norm_method == 'pre-norm':
            y = model_func(self.norm(x), **model_kwargs)
            x = x + y

        elif self.norm_method == 'post-norm':
            y = model_func(x, **model_kwargs)
            x = self.norm(x + y)

        elif self.norm_method == 'pre+post-norm':
            y = model_func(self.pre_norm(x), **model_kwargs)
            x = self.post_norm(x + y)

        elif self.norm_method in ['hypersphere-interpolation', 'hypersphere-spherical-interpolation', 'adaptive-hypersphere-interpolation']:
            y = model_func(x, **model_kwargs)
            # y = torch.nn.functional.normalize(y, p=2, dim=-1) # normalize to hypersphere (unit-norm)

            # # x = torch.lerp(x, y, self.lerp_weight * self.forward_lerp_weight_scale) # interpolate between x and y = func(x)
            # x = x + (self.lerp_weight * self.forward_lerp_weight_scale) * (y - x) # interpolate between x and y = func(x)
            # x = torch.nn.functional.normalize(x, p=2, dim=-1) # normalize to hypersphere (unit-norm)
            x = self.res_stream(x, y)
        else:
            raise ValueError(f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}')

        return x

class HypersphereLERP(nn.Module):
    """
    Implements linear interpolation on the hypersphere, based on the nGPT paper:
    "Normalized Transformer with Representation Learning on the Hypersphere" (arxiv.org/abs/2410.01131).

    The basic idea is to maintain embeddings on the unit-norm hypersphere and update them by interpolating
    along geodesics on the hypersphere. This class approximates spherical interpolation (SLERP) as
    linear interpolation (LERP), as proposed in the nGPT paper.

    SLERP(a, b; alpha) = sin((1-alpha) * theta) / sin(theta) * a + sin(alpha * theta) / sin(theta) * b,
    where theta is the angle between a and b = arccos(<a, b>), and alpha is the interpolation weight in [0, 1].

    Here, following the nGPT paper, we approximate this by:
    LERP(a, b; alpha) = a + alpha * (b - a) = (1 - alpha) * a + alpha * b.
    In the limit as theta -> 0, SLERP(a, b; alpha) -> LERP(a, b; alpha).

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    lerp_scale : float, optional
        Scale factor for the interpolation weight. Default is sqrt(dim).
    lerp_init : float, optional
        Initial value for the interpolation weight. Default is 1.0.
    lerp_weight_constraint : str, optional
        Constraint to apply to the interpolation weight. Options are 'none', 'sigmoid', 'abs', 'clamp'. Default is 'none'.

    Methods
    -------
    forward(x, y)
        Performs the linear interpolation on the hypersphere between tensors x and y.

    """

    def __init__(self, dim, lerp_scale=None, lerp_init=1.0, lerp_weight_constraint='none'):
        super().__init__()

        self.dim = dim
        self.lerp_init = lerp_init
        self.lerp_scale = lerp_scale if lerp_scale is not None else self.dim ** 0.5
        self.lerp_weight = nn.Parameter(torch.ones(self.dim) * self.lerp_scale, requires_grad=True)
        self.forward_lerp_weight_scale = self.lerp_init / self.lerp_scale

        # if normalize_lerp_weight, then normalize lerp_weight to [0,1] using sigmoid
        # NOTE: in nGPT paper, they don't normalize interpolation weight alpha
        # (which is a bit confusing to me, since operation is not interpretable and may be strongly biased to)
        self.lerp_weight_constraint = lerp_weight_constraint # whether to normalize lerp_weight to [0,1]
        assert lerp_weight_constraint in ['none', 'sigmoid', 'abs', 'clamp']
        self.lerp_weight_constraint_fn = {
            'none': lambda x: x,
            'sigmoid': lambda x: x.sigmoid(),
            'abs': lambda x: torch.abs(x),
            'clamp': lambda x: x.clamp(0, 1),
        }.get(lerp_weight_constraint)

    def forward(self, x, y):
        # normalize/project to hypersphere
        # typically (e.g., in ResNet architecture with this resstream method), x will already be normalized to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)

        interpolation_weight = self.lerp_weight_constraint_fn(self.lerp_weight * self.forward_lerp_weight_scale)
        x = x + interpolation_weight * (y - x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

class HypersphereSLERP(nn.Module):
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

    def __init__(self, dim, single_weight=True):
        super().__init__()

        self.dim = dim
        self.single_weight = single_weight

        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight = nn.Parameter(torch.ones(1) if single_weight else torch.ones(self.dim), requires_grad=True)
        # what is geometric interpretation of single_weight = False?

    def forward(self, x, y):
        # x, y: [batch_size, ..., dim]

        # normalize to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight.sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * y
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

class AdaptiveHypersphereSLERP(nn.Module):
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

    def __init__(self, dim, single_weight=True):
        super().__init__()

        self.dim = dim
        self.single_weight = single_weight

        # linear map from y to interpolation weight alpha
        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight_map = nn.Linear(dim, 1) if single_weight else nn.Linear(dim, dim)

    def forward(self, x, y):
        # x, y: [batch_size, ..., dim]

        # normalize to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight_map(y).sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * y
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

    # TODO: IDEA: alpha interpolation weight can be learnable function of x and/or y. This implements a natural type of gating mechanism.

class ConcatCombine(nn.Module):
    def __init__(self, dim, dim2=None):
        super().__init__()
        self.dim = dim
        self.dim2 = dim2 if dim2 is not None else dim
        self.total_dim = self.dim + self.dim2
        self.combine = nn.Linear(self.total_dim, self.dim, bias = False)

    def forward(self, x, skip):
        concatted_skip = torch.cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

def create_norm(d_model, norm_type):
    if norm_type=='layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type=='rmsnorm':
        return nn.RMSNorm(d_model)
    elif norm_type == 'l2':
        return partial(torch.nn.functional.normalize, dim=-1, p=2)
    elif norm_type=='none':
        return  nn.Identity()
    else:
        raise ValueError(f'norm_type {norm_type} not valid')

# endregion