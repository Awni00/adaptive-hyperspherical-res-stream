import torch
import pytorch_lightning as pl
import lightning.pytorch.utilities
from datetime import datetime

from models.recurrent_llama import RecurrentTransformer as Llama
from models.recurrent_llama import ModelArgs as LlamaArgs
from models.recurrent_nGPT import RecurrentnGPT
from utils.utils import get_cosine_schedule_with_warmup, format_large_number, AttributeDict

class LitRecurrentTransformerLM(pl.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.model = create_model(model_config)
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Compile model: ', self.train_config.get('compile', False))
        if self.train_config.get('compile', False):
            self.model = torch.compile(self.model)

        print('Use AMP:', train_config.get('amp', False))
        self.ctx_manager = torch.amp.autocast(enabled=(train_config.get('amp', False)), dtype=torch.bfloat16, device_type='cuda')


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with self.ctx_manager:
            logits = self.model(x)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        self.log('train/loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/ppl', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        self.log('train/tokens', batch_idx * x.size(0) * x.size(1), on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        self.log('val/loss', loss, prog_bar=True, logger=True)
        self.log('val/ppl', torch.exp(loss), prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Configure the optimizer.
        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

        optimizer_name = self.train_config.optimizer
        if optimizer_name not in optimizer_dict.keys():
            raise ValueError(f"Optimizer {optimizer_name} is not implemented!")
        else:
            optimizer = optimizer_dict[optimizer_name](
                self.parameters(),
                **self.train_config[f'{optimizer_name}_optimizer_config']
            )

        # Configure the learning rate scheduler.
        if self.train_config.lr_scheduler == "cosine":
            cosine_scheduler_config = self.train_config.cosine_scheduler_config
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                max_lr=self.train_config[f'{optimizer_name}_optimizer_config']['lr'],
                lr_decay_steps=cosine_scheduler_config.get('lr_decay_steps', self.train_config.n_train_steps),
                min_lr=cosine_scheduler_config.get('min_lr', None),
                warmup_iters=cosine_scheduler_config.get('warmup_steps', None),
            )

        elif self.train_config.lr_scheduler == "step":
            StepLR_config = self.train_config.StepLR_scheduler_config
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=StepLR_config.step_size,
                gamma=StepLR_config.gamma,
            )

        else:
            # use no scheduler
            scheduler = None

        # if using manual_norm_weights = True (i.e., parametrize=False in NormLinear layer), parameterization does not enforce norm constraint
        # instead, we manually normalize the weights after each optimizer step
        if self.model_config.get('manual_norm_weights', False):
            self.model.register_step_post_hook(optimizer)

        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
    # NOTE: configure_optimizer in models.language_models uses slightly different config for tensors of different ranke

    def lr_scheduler_step(
            self,
            scheduler,
            metric,
    ) -> None:
        scheduler.step()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        This function is called before the optimizer step.
        You can override this function to do something before the optimizer step.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer
        """
        norms = lightning.pytorch.utilities.grad_norm(self.model, norm_type=2)
        self.log_dict(norms)


def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Model Config
    # Name: Seed + Date-Time
    model_str = f'{model_config.model_type} - L{model_config.n_layers}T{model_config.n_iters}H{model_config.n_heads}D{model_config.d_model}'

    if model_config.model_type == 'nGPT':
        model_str += f' - {model_config.residual_module}'

        if model_config.residual_module in ['ResidualSphericalSLERP', 'ResidualAdaptiveSphericalSLERP']:
            model_str += f" - SW-{model_config.residual_module_kwargs['single_weight']}"
        if 'n_spheres' in model_config.get('residual_module_kwargs', {}):
            model_str += f" - NS-{model_config.residual_module_kwargs['n_spheres']}"
        if 'slerp_weight_map' in model_config.get('residual_module_kwargs', {}):
            model_str += f" - SWM-{model_config.residual_module_kwargs['slerp_weight_map']}"
        if 'interpolation_weight_activation' in model_config.get('residual_module_kwargs', {}):
            model_str += f" - IWAct-{model_config.residual_module_kwargs['interpolation_weight_activation']}"
        if "manual_norm_weights" in model_config:
            model_str += f" - MNW-{model_config.manual_norm_weights}"

    data_str = f'{data_config.sequence_length}'

    group_name = f'{model_str} - {data_str}' #  - {train_str} - {data_str}

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name

def parse_model_config(model_config):
    # remove non-applicable keys
    if model_config['model_type'] == 'llama':
        model_config.pop('residual_module', None)
        model_config.pop('residual_module_kwargs', None)
        model_config.pop('manual_norm_weights', None)
        model_config.pop('attn_norm_qk', None)
        model_config.pop('num_hyperspheres', None)
        model_config.pop('add_value_residual', None)

    if model_config['model_type'] == 'nGPT':
        if model_config['residual_module'] == 'SphericalLERP':
            model_config.pop('residual_module_kwargs', None)

    return model_config


def create_model(model_config):
    if model_config['model_type'] == 'nGPT':
        nGPT_config = dict(
            vocab_size=model_config['vocab_size'],

            dim=model_config['d_model'],
            depth=model_config['n_layers'],
            n_iters=model_config['n_iters'],
            dim_head=model_config['d_model'] // model_config['n_heads'],
            heads=model_config['n_heads'],
            tied_embedding=model_config['tied_embedding'],

            # residual module
            residual_module=model_config.get('residual_module', 'SphericalLERP'),
            residual_module_kwargs=model_config.get('residual_module_kwargs', None),

            # parameterization of NormLinear
            manual_norm_weights=model_config.get('manual_norm_weights', False),

            attn_norm_qk=model_config.get('attn_norm_qk', True), # whether to normalize q and k after {q,k} = x W_{q,k}
            ff_expand_factor=4, # fixed
            ce_ignore_index=-1,
            num_hyperspheres=model_config.get('num_hyperspheres', 1),
            causal=True,
            add_value_residual=model_config.get('add_value_residual', False), # this is based on https://arxiv.org/abs/2410.17897v1
            )

        print('-'*50)
        print('nGPT config')
        print(AttributeDict(nGPT_config))
        print('-'*50)

        model = RecurrentnGPT(**nGPT_config)

    elif model_config['model_type'] == 'llama':
        llama_config = LlamaArgs(
            dim=model_config['d_model'],
            n_layers = model_config['n_layers'],
            n_iters = model_config['n_iters'],
            n_heads = model_config['n_heads'],
            tied_embedding=model_config['tied_embedding'],

            n_kv_heads = None,
            vocab_size = model_config['vocab_size'],
            multiple_of = 256,  # make SwiGLU hidden layer size multiple of large power of 2
            ffn_dim_multiplier = None,
            norm_eps = 1e-5,
            rope_theta = 500000,
            use_scaled_rope = False,
            max_batch_size = 32,
            max_seq_len = 2048,
            flash = True, # use flash attention?
        )

        print('-'*50)
        print('Llama config')
        print(AttributeDict(vars(llama_config)))
        print('-'*50)

        model = Llama(llama_config)
    else:
        raise ValueError(f'Invalid model type: {model_config["model_type"]}')

    return model