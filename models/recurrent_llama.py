import os
import glob
# import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .llama import TransformerBlock, RMSNorm, precompute_freqs_cis

# -----------------------------------------------------------------------------
# ModelArgs

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_iters: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    tied_embedding: bool = True
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # use flash mention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class RecurrentTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_iters = params.n_iters

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        if params.tied_embedding:
            self.output.weight = self.tok_embeddings.weight

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        # for use during inference
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for _ in range(self.n_iters):
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor=None, ignore_index=-100):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first

        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for _ in range(self.n_iters):
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h)

        if targets is None:
            return logits

        # and then loss
        loss = F.cross_entropy(
            input=logits.float().transpose(1, 2),
            target=targets,
            reduction="mean",
            ignore_index=ignore_index,
        )
        return loss

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor=None, ignore_index=-100):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for _ in range(self.n_iters):
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h)

        if targets is None:
            return logits

        # and then loss
        loss = F.cross_entropy(
            input=logits.float().transpose(1, 2),
            target=targets,
            reduction="mean",
            ignore_index=ignore_index,
        )
        return loss

    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        train_params = []

        finetune_type = "all"
        if finetune_type == "rmsnorm":
            # let's only train the RMSNorm parameters to start
            for name, param in self.named_parameters():
                if "norm" in name:
                    train_params.append(param)
        elif finetune_type == "all":
            # let's train all parameters
            for param in self.parameters():
                train_params.append(param)
        elif finetune_type == "all_no_pos":
            # let's train all parameters except the positional embeddings and lm_head
            n, m = 0, 0
            for name, param in self.named_parameters():
                if name == "output.weight":
                    # do not include
                    n += 1
                    continue
                elif name == "tok_embeddings.weight":
                    # do not include and also does not require grad
                    m += 1
                    param.requires_grad = False
                else:
                    # do include
                    train_params.append(param)
            assert n == 1, "did not find output.weight"
            assert m == 1, "did not find tok_embeddings.weight"

        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("number of trainable parameters: ", sum(p.numel() for p in train_params))
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(train_params, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

# # -----------------------------------------------------------------------------
# # Llama wrapper

# class Llama:

#     @staticmethod
#     def build(
#         ckpt_dir: str,
#         tokenizer_path: str,
#         max_seq_len: int,
#         max_batch_size: int,
#         flash: bool = False,
#         model_parallel_size: Optional[int] = 1,
#         seed: int = 1,
#     ) -> "Llama":
#         assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
#         assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
#         assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

#         local_rank = 0
#         torch.cuda.set_device(local_rank)
#         torch.manual_seed(seed) # seed must be the same in all processes

#         start_time = time.time()
#         checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
#         assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
#         assert model_parallel_size == len(checkpoints)
#         ckpt_path = checkpoints[0]
#         checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
#         with open(Path(ckpt_dir) / "params.json", "r") as f:
#             params = json.loads(f.read())

#         model_args: ModelArgs = ModelArgs(
#             max_seq_len=max_seq_len,
#             max_batch_size=max_batch_size,
#             flash=flash,
#             **params,
#         )
#         tokenizer = Tokenizer(model_path=tokenizer_path)
#         assert model_args.vocab_size == tokenizer.n_words
#         if torch.cuda.is_bf16_supported():
#             torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
#         else:
#             torch.set_default_tensor_type(torch.cuda.HalfTensor)
#         model = Transformer(model_args)
#         model.load_state_dict(checkpoint, strict=False)
#         print(f"Loaded in {time.time() - start_time:.2f} seconds")
#         return Llama(model, tokenizer)

#     def __init__(self, model: Transformer, tokenizer: Tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     @torch.inference_mode()
#     def generate(
#         self,
#         prompt_tokens: List[List[int]],
#         sample_rng: torch.Generator,
#         max_gen_len: int,
#         temperature: float = 0.6,
#         top_p: float = 0.9,
#         echo: bool = False,
#     ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
#         """
#         prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
#         max_gen_len (int): Maximum length of the generated text sequence.
#         temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
#         top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
#         logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
#         echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
#         """
#         params = self.model.params
#         bsz = len(prompt_tokens)
#         assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

#         min_prompt_len = min(len(t) for t in prompt_tokens)
#         max_prompt_len = max(len(t) for t in prompt_tokens)
#         assert max_prompt_len <= params.max_seq_len
#         total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

#         # install KV cache in all the Attention layers
#         for block in self.model.layers:
#             layer_dtype = block.attention.wq.weight.dtype
#             layer_device = block.attention.wq.weight.device
#             block.attention.cache = KVCache(
#                 batch_size=bsz,
#                 seq_length=total_len,
#                 n_kv_heads=params.n_kv_heads,
#                 head_dim=params.dim // params.n_heads,
#                 dtype=layer_dtype,
#                 device=layer_device,
#             )

#         pad_id = self.tokenizer.pad_id
#         tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
#         for k, t in enumerate(prompt_tokens):
#             tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

#         prev_pos = 0
#         eos_reached = torch.tensor([False] * bsz, device="cuda")
#         input_text_mask = tokens != pad_id

#         if min_prompt_len == total_len:
#             logits = self.model.forward_inference(tokens, prev_pos)

#         stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

#         for cur_pos in range(min_prompt_len, total_len):
#             # get the logits for the next token in all the batch rows
#             logits = self.model.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)
#             # sample the next token
#             if temperature > 0:
#                 probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
#                 next_token = sample_top_p(probs, top_p, sample_rng)
#             else:
#                 next_token = torch.argmax(logits[:, -1], dim=-1)
#             next_token = next_token.reshape(-1)
#             # only replace token if prompt has already been generated
#             next_token = torch.where(
#                 input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
#             )
#             tokens[:, cur_pos] = next_token
#             eos_reached |= (~input_text_mask[:, cur_pos]) & (
#                 torch.isin(next_token, stop_tokens)
#             )
#             prev_pos = cur_pos
#             if all(eos_reached):
#                 break

#         out_tokens = []
#         for i, toks in enumerate(tokens.tolist()):
#             # cut to max gen len
#             start = 0 if echo else len(prompt_tokens[i])
#             toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
#             # cut to after eos tok if any
#             for stop_token in self.tokenizer.stop_tokens:
#                 try:
#                     eos_idx = toks.index(stop_token)
#                     toks = toks[:eos_idx]
#                 except ValueError:
#                     pass
#             out_tokens.append(toks)

#         # clean up the KV cache in all the layers
#         for block in self.model.layers:
#             block.attention.cache = None

#         return out_tokens

#     def text_completion(
#         self,
#         prompts: List[str],
#         sample_rng: torch.Generator,
#         temperature: float = 0.6,
#         top_p: float = 0.9,
#         max_gen_len: Optional[int] = None,
#         echo: bool = False,
#     ):
#         if max_gen_len is None:
#             max_gen_len = self.model.params.max_seq_len - 1
#         # encode the (string) prompts to tokens
#         prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
#         # generate the completions in tokens space
#         generation_tokens = self.generate(
#             prompt_tokens=prompt_tokens,
#             sample_rng=sample_rng,
#             max_gen_len=max_gen_len,
#             temperature=temperature,
#             top_p=top_p,
#             echo=echo,
#         )
#         # decode the completions back to strings
#         completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
#         return completions

# def sample_top_p(probs, p, generator):
#     probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
#     probs_sum = torch.cumsum(probs_sort, dim=-1)
#     mask = probs_sum - probs_sort > p
#     probs_sort[mask] = 0.0
#     probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
#     next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
#     next_token = torch.gather(probs_idx, -1, next_token)
#     return next_token

# # -----------------------------------------------------------------------------
# # distributed and sharded data loader

# def _peek_data_shard(filename):
#     # only reads the header, returns header data
#     with open(filename, "rb") as f:
#         # first read the header, which is 256 int32 integers (4 bytes each)
#         header = np.frombuffer(f.read(256*4), dtype=np.int32)
#     if header[0] != 20240801:
#         print("ERROR: magic number mismatch in the data .bin file!")
#         exit(1)
#     assert header[1] == 7, "unsupported version"
#     ntok = header[2] # number of tokens (claimed)
#     return ntok # for now just return the number of tokens

# def _load_data_shard(filename):
#     with open(filename, "rb") as f:
#         # first read the header, which is 256 int32 integers (4 bytes each)
#         header = np.frombuffer(f.read(256*4), dtype=np.int32)
#         assert header[0] == 20240801, "magic number mismatch in the data .bin file"
#         assert header[1] == 7, "unsupported version"
#         ntok = header[2] # number of tokens (claimed)
#         # the rest of it are tokens, stored as uint16
#         tokens = np.frombuffer(f.read(), dtype=np.uint32)
#     assert len(tokens) == ntok, "number of tokens read does not match header?"
#     return tokens

# class DistributedShardedDataLoader:
#     """
#     This DataLoader is both:
#     - distributed (works correctly in case of multiple processes in DDP)
#     - sharded (supports datasets that are broken up into multiple data shards)
#     It is not *permuted*, meaning that it itearates over the data in the order
#     of the dataset on disk, so the user should make sure to shuffle their examples
#     during the creation of their data shards for best performance.
#     """
#     def __init__(self, filename_pattern, B, T, process_rank, num_processes):
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         self.B = B
#         self.T = T

#         # glob files that match the pattern
#         self.files = sorted(glob.glob(filename_pattern))
#         assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

#         # load and validate all data shards, count number of tokens in total
#         ntok_total = 0
#         for fname in self.files:
#             shard_ntok = _peek_data_shard(fname)
#             assert shard_ntok >= num_processes * B * T + 1
#             ntok_total += shard_ntok
#         self.ntok_total = ntok_total
#         print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

#         # kick things off
#         self.current_shard = None
#         self.reset()

#     def reset(self):
#         # we're being a bit clever here: if we already had shard 0 loaded,
#         # then don't do the work to reload it, just reset the pointer
#         if self.current_shard != 0:
#             self.current_shard = 0
#             self.tokens = _load_data_shard(self.files[self.current_shard])
#         self.current_position = self.process_rank * self.B * self.T

#     def advance(self): # advance to next data shard
#         self.current_shard = (self.current_shard + 1) % len(self.files)
#         self.current_position = self.process_rank * self.B * self.T
#         self.tokens = _load_data_shard(self.files[self.current_shard])

#     def next_batch(self):
#         B = self.B
#         T = self.T
#         buf = self.tokens[self.current_position : self.current_position+B*T+1]
#         buf = torch.tensor(buf, dtype=torch.long)
#         x = (buf[:-1]).view(B, T) # inputs
#         y = (buf[1:]).view(B, T) # targets
#         # advance the start pointer in current shard
#         self.current_position += B * T * self.num_processes
#         # if loading the next batch would be out of bounds advance the shard
#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             self.advance()
#         return x, y
