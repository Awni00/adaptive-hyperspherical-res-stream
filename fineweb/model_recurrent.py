import torch
import pytorch_lightning as pl
import lightning.pytorch.utilities
from datetime import datetime

from models.recurrent_llama import RecurrentTransformer as Llama
from models.recurrent_llama import ModelArgs as LlamaArgs
from models.recurrent_nGPT import RecurrentnGPT
from models.transformer_baseline import RecurrentTransformerLM as BaselineRecurrentTransformer
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
        ngpt_config = model_config['nGPT_kwargs']

        model_str += f' - {ngpt_config.residual_module}'
        if ngpt_config.residual_module in ['ResidualSphericalSLERP', 'ResidualAdaptiveSphericalSLERP']:
            model_str += f" - SW-{ngpt_config.residual_module_kwargs['single_weight']}"
        if 'n_spheres' in ngpt_config.get('residual_module_kwargs', {}):
            model_str += f" - NS-{ngpt_config.residual_module_kwargs['n_spheres']}"
        if 'slerp_weight_map' in ngpt_config.get('residual_module_kwargs', {}):
            model_str += f" - SWM-{ngpt_config.residual_module_kwargs['slerp_weight_map']}"
            if ngpt_config.get('residual_module_kwargs', {}).get('bias', None):
                model_str += f"Bias"
        if 'interpolation_weight_activation' in ngpt_config.get('residual_module_kwargs', {}):
            model_str += f" - IWAct-{ngpt_config.residual_module_kwargs['interpolation_weight_activation']}"
        if "manual_norm_weights" in ngpt_config:
            model_str += f" - MNW-{ngpt_config.manual_norm_weights}"

    if model_config.model_type == 'baseline_transformer':
        baseline_kwargs = model_config.get('baseline_transformer_kwargs', {})
        model_str += f' - GPTInit-{baseline_kwargs.gpt_special_init}'

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
    # remove non-applicable
    for model_type in ['nGPT', 'llama', 'baseline_transformer']:
        if model_config['model_type'] != model_type:
            model_config.pop(f'{model_type}_kwargs', None)

    return model_config


def create_model(model_config):
    if model_config['model_type'] == 'nGPT':
        ngpt_kwargs = model_config.get('nGPT_kwargs', {})
        nGPT_config = dict(
            vocab_size=model_config['vocab_size'],

            dim=model_config['d_model'],
            depth=model_config['n_layers'],
            n_iters=model_config['n_iters'],
            dim_head=model_config['d_model'] // model_config['n_heads'],
            heads=model_config['n_heads'],
            tied_embedding=model_config['tied_embedding'],

            # residual module
            residual_module=ngpt_kwargs.get('residual_module', 'SphericalLERP'),
            residual_module_kwargs=ngpt_kwargs.get('residual_module_kwargs', None),

            # parameterization of NormLinear
            manual_norm_weights=ngpt_kwargs.get('manual_norm_weights', False),
            attn_norm_qk=ngpt_kwargs.get('attn_norm_qk', True), # whether to normalize q and k after {q,k} = x W_{q,k}
            num_hyperspheres=ngpt_kwargs.get('num_hyperspheres', 1),
            add_value_residual=ngpt_kwargs.get('add_value_residual', False), # this is based on https://arxiv.org/abs/2410.17897v1

            # fixed
            ff_expand_factor=4,
            ce_ignore_index=-1,
            causal=True,
            )

        print('-'*50)
        print('nGPT config')
        print(AttributeDict(nGPT_config))
        print('-'*50)

        model = RecurrentnGPT(**nGPT_config)

    elif model_config['model_type'] == 'llama':
        llama_config = LlamaArgs(
            vocab_size = model_config['vocab_size'],

            dim=model_config['d_model'],
            n_layers = model_config['n_layers'],
            n_iters = model_config['n_iters'],
            n_heads = model_config['n_heads'],
            tied_embedding=model_config['tied_embedding'],

            n_kv_heads = None,
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

    elif model_config['model_type'] == 'baseline_transformer':
        baseline_kwargs = model_config.get('baseline_transformer_kwargs', {})
        transformer_config = AttributeDict(
            vocab_size = model_config['vocab_size'],

            d_model = model_config['d_model'],
            n_heads = model_config['n_heads'],
            dff = model_config['d_model'] * 4,
            n_layers = model_config['n_layers'],
            n_iters = model_config['n_iters'],
            tied_embedding = model_config['tied_embedding'],

            mlp_activation = baseline_kwargs.get('mlp_activation', 'swiglu'),
            pos_enc_type = baseline_kwargs.get('pos_enc_type', 'rotary'),
            norm_config = baseline_kwargs.norm_config,
            bias = baseline_kwargs.get('bias', False),
            gpt_special_init = baseline_kwargs.get('gpt_special_init', False),
            )

        print('-'*50)
        print('Baseline Transformer config')
        print(AttributeDict(transformer_config))
        print('-'*50)

        model = BaselineRecurrentTransformer(transformer_config)
    else:
        raise ValueError(f'Invalid model type: {model_config["model_type"]}')

    return model