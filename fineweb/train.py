import argparse
import yaml
import ast
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchinfo
import tiktoken

import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from fineweb_data import FinewebDataset

import os, sys; sys.path.insert(0, os.path.abspath('..')) # add project root dir to path
from utils.utils import AttributeDict, print_gpu_info, format_large_number

from model import LitTransformerLM, get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, default='data/dag_data.pt')
parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging, no checkpoints).')

args, unknown_args = parser.parse_known_args()

# load model, train, and data config
with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
    model_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
    train_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
    data_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

# update configs with command line arguments (if applicable)
if len(unknown_args) > 0:
    print('='*80)
    print("Received unknown arguments:", unknown_args)
for arg_str in unknown_args:
    for (prefix, config) in [('--train_config.', train_config), ('--data_config.', data_config), ('--model_config.', model_config)]:
        if arg_str.startswith(prefix):
            key, value = arg_str[len(prefix):].split('=')
            value = ast.literal_eval(value)
            key_parts = key.split('.')
            # traverse through the config dict to the second last key
            for k in key_parts[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[key_parts[-1]] = value
            print(f"Updated {prefix}{key} to {value}")

# process defaults
train_config.val_check_interval = getattr(train_config, 'val_check_interval', 250) # interval (in microbatches) to check validation
train_config.limit_val_batches = getattr(train_config, 'limit_val_batches', 20) # number of microbatches to use for validationtrain_config.
train_config.log_every_n_steps = getattr(train_config, 'log_every_n_steps', None)
train_config.gradient_clip_val = getattr(train_config, 'gradient_clip_val', None)
train_config.max_time = getattr(train_config, 'max_time', None) # will stop training after this amount of time
train_config.benchmark = getattr(train_config, 'benchmark', False)
train_config.profiler = getattr(train_config, 'profiler', None)
train_config.limit_train_batches = getattr(train_config, 'limit_train_batches', None) # number of batches to use for training (useful for debugging)

# setup
tokenizer = tiktoken.get_encoding('gpt2')
model_config.vocab_size = tokenizer.n_vocab

# print configs
print('='*80)
print("Model Config:")
print(model_config)
print('-'*80)
print("Train Config:")
print(train_config)
print('-'*80)
print("Data Config:")
print(data_config)
print('='*80)

# GPU diagnostics
print_gpu_info()

# current device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set the seed for reproducibility
seed = np.random.randint(0, 2**32) # randomly sample a seed
pl.seed_everything(seed) # sets the seed for all random number generators
train_config.seed = seed
print('='*80)

# load data
train_config.batch_size = train_config.tokens_per_batch // data_config.sequence_length
train_config.micro_batch_size = train_config.get('micro_batch_size', train_config.batch_size) # set micro batch size, if not set
train_config.gradient_accumulation_steps = train_config.batch_size // train_config.micro_batch_size # set grad accum steps

train_data = FinewebDataset(sequence_length=data_config.sequence_length, split='train')
val_data = FinewebDataset(sequence_length=data_config.sequence_length, split='val')

train_loader = DataLoader(train_data, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=4, pin_memory=True)

if train_config.log_every_n_steps is None:
    train_config.log_every_n_steps = 1 # max(train_config.gradient_accumulation_steps, 1) # FIXME

# dynamically choose micro batch size, gradient accumulation steps, number of training steps, etc...

n_train_microsteps = len(train_loader)
train_config.n_train_microsteps = n_train_microsteps # used by the learning rate scheduler
train_config.n_train_steps = n_train_microsteps // train_config.gradient_accumulation_steps

if train_config.get('limit_train_batches', None) is not None:
    train_config.n_train_steps = min(train_config.n_train_steps, train_config.limit_train_batches)

print('='*80)
print(f'Total # of Training Tokens: {format_large_number(train_data.num_tokens)}')
print(f'Tokens per batch: {train_config.tokens_per_batch:,}')
print(f'Batch size: {train_config.batch_size}')
print(f'Micro batch size: {train_config.micro_batch_size}')
print(f'Gradient accumulation steps: {train_config.gradient_accumulation_steps}')
print(f'Total # of Training Steps: {train_config.n_train_steps:,} steps; {train_config.n_train_microsteps:,} microsteps')
print('='*80)

# model
print('='*80)
print('Initializing model...')
litmodel = LitTransformerLM(model_config, train_config)#.to(device)

print(litmodel.model)

# get torchinfo summary
print()

model_summary = torchinfo.summary(litmodel.model, input_data=torch.zeros((train_config.micro_batch_size, data_config.sequence_length), dtype=torch.long),
    col_names=("input_size", "output_size", "num_params", "params_percent"))

model_summary_dict = AttributeDict({
    'Input size (MB)': model_summary.to_megabytes(model_summary.total_input),
    'Params size (MB)': model_summary.to_megabytes(model_summary.total_param_bytes),
    'Forward/backward pass size  (MB)': model_summary.to_megabytes(model_summary.total_output_bytes),
    'Estimated total size (MB)': model_summary.to_megabytes(model_summary.total_output_bytes + model_summary.total_param_bytes + model_summary.total_input),
    'Total Mult-Adds': model_summary.total_mult_adds,

    'trainable_params': model_summary.trainable_params, # note: numbers from torchinfo are not always accurate
    'total_params': model_summary.total_params, # note: numbers from torchinfo are not always accurate

    'num_params': sum(p.numel() for p in litmodel.model.parameters()),
    'num_trainable_params': sum(p.numel() for p in litmodel.model.parameters() if p.requires_grad)
    })

print('='*80)

# logger

experiment_config = dict(train_config=train_config, model_config=model_config, data_config=data_config, model_summary=model_summary_dict)

# initialize wandb run
# Group: Data Config - Model Config / Run name: Seed + Date-Time
group_name, run_name = get_experiment_name(model_config, data_config, train_config)

train_config.experiment_run_name = run_name
train_config.experiment_group = group_name

if not args.debug:
    wandb_experiment_run = wandb.init(
        entity=train_config.wandb_config.wandb_entity, project=train_config.wandb_config.wandb_project,
        name=train_config.experiment_run_name, group=train_config.experiment_group, config=experiment_config)

# configure callbacks
callbacks = []
if not args.debug:

    logger = pl.loggers.WandbLogger(
        experiment=wandb_experiment_run,
        config=experiment_config,
        log_model=train_config.wandb_config.get('log_model', False)
        )

    if getattr(train_config.wandb_config, 'watch', False):
        logger.watch(litmodel.model, log='all', log_graph=True)
else:
    logger = None

# callbacks: checkpoint and lr monitor
if train_config.get('checkpointing', False):
    checkpoint_dir = f'checkpoints/{train_config.experiment_group}-{train_config.experiment_run_name}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{step}-{val/loss:.4f}',
        monitor='val/loss', # this depends on logging in the LightningModule
        mode='min',
    )
    callbacks.append(checkpoint_callback)

if not args.debug:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progbar = pl.callbacks.TQDMProgressBar(refresh_rate=250)
    callbacks.append(lr_monitor)
    callbacks.append(progbar)

# set matmul precision
if getattr(train_config, 'matmul_precision', None) is not None:
    torch.set_float32_matmul_precision(train_config.matmul_precision)
    print(f'Matmul precision set to {train_config.matmul_precision}.')

# configure trainer

trainer_kwargs = dict(
    max_epochs=train_config.get('max_epochs', 1),
    logger=logger,
    callbacks=callbacks,

    val_check_interval=getattr(train_config, 'val_check_interval', 250), # interval to check validation
    limit_val_batches=getattr(train_config, 'limit_val_batches', 20), # number of batches to use for validation

    log_every_n_steps=getattr(train_config, 'log_every_n_steps', None),

    accumulate_grad_batches=train_config.gradient_accumulation_steps, # if micro_batch_size is set, this is set to batch_size // micro_batch_size
    gradient_clip_val=getattr(train_config, 'gradient_clip_val', None),

    max_time=getattr(train_config, 'max_time', None),
    benchmark=getattr(train_config, 'benchmark', False),
    profiler=getattr(train_config, 'profiler', None),
    limit_train_batches=getattr(train_config, 'limit_train_batches', None), # number of batches to use for training (useful for debugging)
)

# note: all *_batches arguments in trainer correspond to microbatches (i.e., counting each microbatch in gradient accumulation as a separate batch)
# we will scale the number of batches by the gradient accumulation steps to get the number of actual (mini)batches
for k in ['val_check_interval', 'limit_val_batches', 'limit_train_batches']:
    # if set as integer, scale the number of batches by the gradient accumulation steps
    if k in trainer_kwargs and isinstance(trainer_kwargs[k], int):
        trainer_kwargs[k] *= train_config.gradient_accumulation_steps
    # if set as a float, we already have the correct behavior

# NOTE: if using a distributed strategy, PL will automatically wrap the model in a `torch.utils.data.DistributedSampler` and set shuffle=True for the train loader
# we don't want to shuffle since that slows down the data loading significantly. If using a distributed strategy, set use_dist_sampler=False in the trainer config
# and manually add our own distributed sampler in the dataloader hooks. (or leave use_dist_sampler=True, PL shouldn't replace sampler it if already added)

trainer = Trainer(**trainer_kwargs)

print('-'*80)
print('Trainer kwargs')
print(AttributeDict(trainer_kwargs))
print('-'*80)

print('Starting training...')

trainer.fit(litmodel, train_loader, val_loader)


# manual train loop

# model = litmodel.model.to(device)

# # opt_config = dict(**train_config.AdamW_optimizer_config, device_type='cuda')
# # optimizer = configure_optimizers(model, **opt_config)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9) # fixme
# criterion =  torch.nn.CrossEntropyLoss()

# if train_config.get('precision', None) == 'bf16-mixed':
#     print("Using AMP.")

# eval_interval = train_config.get('eval_interval', 250)
# val_loss_steps = train_config.get('val_loss_steps', 250)


# amp_ctx_manager = torch.amp.autocast(device_type=device.type, enabled=train_config.get('amp', False), dtype=torch.bfloat16)
# # train_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
# n_steps = len(train_loader) // train_config.gradient_accumulation_steps
# train_iterator = tqdm(range(n_steps), total=n_steps)

# train_loader = iter(train_loader)
# val_loader = iter(val_loader)

# # note: no learning rate scheduler for now

# for step in train_iterator:

#     optimizer.zero_grad()

#     # gradient accumulation
#     batch_loss = 0
#     for micro_step in range(train_config.gradient_accumulation_steps):
#         x, y = next(train_loader)
#         x, y = x.to(device), y.to(device)

#         with amp_ctx_manager:
#             logits = model(x)
#         loss = criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

#         loss /= train_config.gradient_accumulation_steps
#         batch_loss += loss.detach().item()

#         loss.backward()


#     optimizer.step()

#     # get elapsed time from tqdm iterator
#     elapsed_time = train_iterator.format_dict['elapsed']
#     tokens = step*train_config.batch_size*data_config.sequence_length
#     tokens_per_sec = tokens / elapsed_time

#     train_iterator.set_postfix(loss=batch_loss, step=step,
#         tokens=format_large_number(step*train_config.batch_size*data_config.sequence_length),
#         tokens_per_sec=format_large_number(tokens_per_sec))

#     if step % 100 == 0:
#         if not args.debug:
#             wandb.log({'train_loss': batch_loss, 'tokens': step*train_config.batch_size*data_config.sequence_length}, step=step)
