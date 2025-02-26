import math
import os
import time
import warnings
from copy import deepcopy
from functools import partial
from glob import glob
from types import SimpleNamespace
from typing import Dict

import deepspeed
import torch
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.base_optimizer import DeepSpeedOptimizer
from ema_pytorch import EMA

from src.criterions import CriterionBase, CriterionManager
from src.datasets import DataManager
from src.datasets.modules.data_module_base import DataLoaderX
from src.models import ModelBase, ModelManager
from src.utils.misc import *
from src.utils.progress_logger import *


class DeepspeedTrainerBase:
    registered_name: str
    
    def __init__(
        self,
        cfg: SimpleNamespace,
        loggers: SimpleNamespace,
        ):
        super().__init__()
        self.cfg = cfg
        self.loggers = loggers
        
        # prepare for data
        self.data_manager = DataManager(cfg, loggers)
        train_loader = self.data_manager.build_dataloader(split='train')
        val_loader = self.data_manager.build_dataloader(split='val')
        
        # prepare for model, postprocessor
        self.model_manager = ModelManager(cfg, loggers)
        model_without_ddp = self.model_manager.build_model()
        # postprocessor = model_manager.build_postprocessor()
        postprocessor = None
        
        # prepare for criterion
        self.criterion_manager = CriterionManager(cfg, loggers)
        criterion = self.criterion_manager.build_criterion()
        
        # deepspeed wrapper
        ds_engine, ds_optimizer, _, _ = deepspeed.initialize(
            model=model_without_ddp,
            model_parameters=model_without_ddp.parameters(),
            dist_init_required=False,
            config=ConfigMisc.nested_namespace_to_nested_dict(cfg.trainer.deepspeed),
            )
        
        # prepare for EMA (must be called after the ddp_wrapper to avoid potential problems)
        ema_container = self.model_manager.build_ema(model_without_ddp)
        
        self._prepare_for_training(
            ds_engine=ds_engine,
            ema_container=ema_container,
            postprocessor=postprocessor,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            ds_optimizer=ds_optimizer,
            device=self.model_manager.device,
            )
    
    def _prepare_for_training(
        self,
        ds_engine: DeepSpeedEngine,
        ema_container: EMA,
        postprocessor: None,
        criterion: CriterionBase,
        train_loader: DataLoaderX,
        val_loader: DataLoaderX,
        ds_optimizer: DeepSpeedOptimizer,
        device: torch.device,
        ) -> None:

        self.ds_engine = ds_engine
        self.model_without_ddp: ModelBase = self.ds_engine.module
        self.ema_container = ema_container  # still in train mode (inited in ModelManager)
        self.postprocessor = postprocessor
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ds_optimizer = ds_optimizer
        self.device = device
        self.start_epoch = 1
        self.epoch = self.start_epoch - 1
        self.train_outputs = {}
        self.last_val_metrics = {}
        self.best_val_metrics = {}
        self.train_pbar = None
        self.val_pbar = None
        self.dist_eval = self.cfg.trainer.dist_eval
        
        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)
        
        self.trained_iters = 0
        self.total_epochs = self.cfg.trainer.epochs
        real_total_epochs = getattr(self.cfg.trainer, 'real_epochs', None)
        if real_total_epochs is not None:
            self.total_epochs = real_total_epochs
        self.total_iters = self.total_epochs * self.train_len
        
        self.val_epoch_list = self._get_val_epochs()
          
        self.nn_module_list = [self.model, self.criterion]
        self.training = True
        
        if self.ema_container is not None:
            print(LoggerMisc.block_wrapper('Using EMA model to evaluate. Setting EMA model and criterion to eval mode...', '='))
            self.ema_container.eval()
            self.ema_criterion = deepcopy(self.criterion)
            assert hasattr(self.ema_criterion, 'ema_mode'), 'ema_criterion doesn\'t have ema_mode attribute, which means the criterion is not a CriterionBase instance.'
            self.ema_criterion.set_ema_mode(True)
            self.ema_criterion.eval()
        
        self.checkpoint_last_interval = self.cfg.trainer.checkpoint_last_interval  # save the last checkpoint every {checkpoint_last_interval} epochs (keep latest)
        assert self.checkpoint_last_interval > 0, 'checkpoint_last_interval should be a positive integer.'
        self.checkpoint_keep_interval = self.cfg.trainer.checkpoint_keep_interval  # save the checkpoint every {checkpoint_keep_interval} epochs (keep all)
        if self.checkpoint_keep_interval > 0:
            os.makedirs(os.path.join(self.cfg.info.work_dir, 'checkpoint_keep_storage'), exist_ok=True)
        self.breath_time = self.cfg.trainer.trainer_breath_time  # XXX: avoid cpu being too busy
    
    @property
    def epoch_loop(self):
        return range(self.start_epoch, self.total_epochs + 1)
    
    @property
    def lr_groups(self):
        ## _no_wd ones are paired with normal ones, so no need to collect them here
        return_dict = {}
        for integrated_optimizer in self.integrated_optimizers:
            return_dict.update({f'lr_{integrated_optimizer.identifier}_' + param_group['group_name']: param_group['lr']
                for param_group in integrated_optimizer.param_groups if not param_group['group_name'].endswith('_no_wd')})
        return return_dict
    
    @property
    def wd_groups(self):
        ## _no_wd ones are paired with normal ones, so no need to collect them here
        return_dict = {}
        for integrated_optimizer in self.integrated_optimizers:
            return_dict.update({f'wd_{integrated_optimizer.identifier}_' + param_group['group_name']: param_group['weight_decay']
                for param_group in integrated_optimizer.param_groups if not param_group['group_name'].endswith('_no_wd')})
        return return_dict
    
    def _get_val_epochs(self):
        if self.cfg.trainer.eval_freq <= 0:
            val_epochs = [self.total_epochs]
        else:
            val_epochs = list(range(self.cfg.trainer.eval_freq, self.total_epochs + 1, self.cfg.trainer.eval_freq))
            if val_epochs[-1] != self.total_epochs:
                val_epochs.append(self.total_epochs)
        return val_epochs
    
    def _get_pbar(self):
        # called in "before_all_epochs"
        if DistMisc.is_main_process():
            epoch_finished = self.start_epoch - 1
            train_pbar = LoggerMisc.MultiTQDM(
                total=self.total_iters if self.cfg.info.global_tqdm else self.train_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=epoch_finished * self.train_len,
            )
            train_pbar.set_description_str('Train')
            print('')
            val_pbar = LoggerMisc.MultiTQDM(
                total=len(self.val_epoch_list) * self.val_len if self.cfg.info.global_tqdm else self.val_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=len([x for x in self.val_epoch_list if x <= epoch_finished]) * self.val_len,
            )
            val_pbar.set_description_str('Eval ')
            print('')
            self.train_pbar = train_pbar
            self.val_pbar = val_pbar
    
    def _resume_training(self):
        # called in "before_all_epochs"
        if self.cfg.trainer.resume:
            checkpoint_path = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
            print(LoggerMisc.block_wrapper(f'loading the checkpoint from {checkpoint_path}', '>'))
            assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
            checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            if self.ema_container is not None:
                assert 'ema_container' in checkpoint, 'checkpoint does not contain "ema_container".'
                self.ema_container.load_state_dict(checkpoint['ema_container'])
            for integrated_optimizer in self.integrated_optimizers:
                integrated_optimizer.load_state_dict(checkpoint[f'integrated_optimizer_{integrated_optimizer.identifier}'])
            self.start_epoch = checkpoint['epoch'] + 1
            if DistMisc.is_main_process():
                self.best_val_metrics = checkpoint.get('best_val_metrics', {})
                self.last_val_metrics = checkpoint.get('last_val_metrics', {})
            self.trained_iters = checkpoint['epoch'] * self.train_len
            self.epoch = self.start_epoch - 1  # will be the same as {checkpoint['epoch'] + 1} by doing '+1' in "before_one_epoch"
        else:
            print(LoggerMisc.block_wrapper('New trainer.', '>'))
        print(f'Start from epoch: {self.start_epoch}')
        return self.cfg.trainer.resume
    
    def _load_pretrained_models(self):
        class FakeEMA(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.ema_model = model
        
        def _load_pretrained_model(model_path, pretrain_model_name):
            if self.cfg.trainer.load_from_ema:  # use state_dict[EMA] to load
                checkpoint = torch.load(model_path, map_location='cpu')
                key = 'ema_container'
                assert key in checkpoint, f'checkpoint does not contain "{key}", but "load_from_ema" is True.'
                
                print(f'\nLoading {pretrain_model_name} (key="{key}") from {model_path}')
                state_dict = checkpoint[key]
                state_dict.pop('initted', None)
                state_dict.pop('step', None)
                if self.ema_container is not None:
                    ModelMisc.load_state_dict_with_more_info(
                        self.ema_container,
                        state_dict,
                        strict=False,
                        print_keys_level=2,
                        )
                    print(f'EMA container exists. Process of param & buffer copying:\n\tstate_dict[EMA] -> ema_container -> online_model')
                    self.ema_container.copy_params_from_ema_to_model()
                else:
                    fake_ema_container = FakeEMA(self.model_without_ddp)
                    ModelMisc.load_state_dict_with_more_info(
                        fake_ema_container,
                        state_dict,
                        strict=False,
                        print_keys_level=1,
                        )
                    print(f'No EMA. Process of param & buffer copying:\n\tstate_dict[EMA] -> fake_ema_container -> online_model')
                    del fake_ema_container
            else:  # use state_dict[model] to load
                print(f'\nLoading {pretrain_model_name} (key="model") from {model_path}')
                ModelMisc.load_state_dict_with_more_info(
                    self.model_without_ddp,
                    torch.load(model_path, map_location='cpu')['model'],
                    strict=False,
                    print_keys_level=1,
                    )
                if self.ema_container is not None:
                    print(f'EMA container exists. Process of param & buffer copying:\n\tstate_dict[online] -> online_model -> ema_container')
                    self.ema_container.copy_params_from_model_to_ema()
                else:
                    print(f'No EMA. Process of param & buffer copying:\n\tstate_dict[online] -> online_model')
        
        if getattr(self.cfg.trainer, 'pretrained_models', None) is not None:
            for pretrain_model_name, pretrained_model_path in ConfigMisc.nested_namespace_to_nested_dict(self.cfg.trainer.pretrained_models).items():
                if pretrained_model_path is not None:
                    _load_pretrained_model(pretrained_model_path, pretrain_model_name)
    
    @staticmethod
    def _save_or_update_checkpoint(save_dict, work_dir, epoch_finished, label):
        # label: 'last' or 'best'
        checkpoint_path_list = glob(os.path.join(work_dir, f'checkpoint_{label}_epoch_*.pth'))
        if len(checkpoint_path_list) > 1:
            warnings.warn(f'Found {len(checkpoint_path_list)} {label} checkpoints, please check.')
        max_saved_temp_epoch = max([int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]) for checkpoint_path in checkpoint_path_list] + [0])
        new_path = os.path.join(work_dir, f'checkpoint_{label}_epoch_{epoch_finished}.pth')
        if max_saved_temp_epoch == 0:
            torch.save(save_dict, new_path)
        else:
            old_path = os.path.join(work_dir, f'checkpoint_{label}_epoch_{max_saved_temp_epoch}.pth')
            torch.save(save_dict, old_path)
            os.rename(old_path, new_path)
    
    def _save_checkpoint(self):
        # called in "after_one_epoch"
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            save_last = epoch_finished % self.checkpoint_last_interval == 0
            save_keep = epoch_finished % self.checkpoint_keep_interval == 0 if self.checkpoint_keep_interval > 0 else False
            
            if save_last or save_keep:
                save_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_container': self.ema_container.state_dict() if self.ema_container is not None else None,
                    **{f'integrated_optimizer_{integrated_optimizer.identifier}': integrated_optimizer.state_dict() for integrated_optimizer in self.integrated_optimizers},
                    'last_val_metrics': self.last_val_metrics,
                    'epoch': epoch_finished,
                }
            if save_last:
                self._save_or_update_checkpoint(save_dict, self.cfg.info.work_dir, epoch_finished, 'last')
            if save_keep:
                keep_path = os.path.join(self.cfg.info.work_dir, f'checkpoint_keep_storage/checkpoint_keep_epoch_{epoch_finished}.pth')
                torch.save(save_dict, keep_path)
    
    def _save_checkpoint_only_best_model(self):
        # called in "after_validation"
        if DistMisc.is_main_process():
            self.best_val_metrics, last_is_best = self.criterion.choose_best(
                self.last_val_metrics, self.best_val_metrics
            )
            
            epoch_finished = self.epoch
            
            if last_is_best:
                save_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_container': self.ema_container.state_dict() if self.ema_container is not None else None,
                    'best_val_metrics': self.best_val_metrics,
                    'epoch': epoch_finished,
                }
                self._save_or_update_checkpoint(save_dict, self.cfg.info.work_dir, epoch_finished, 'best')
    
    def _train_mode(self):
        # called in "before_one_epoch"
        for nn_module in self.nn_module_list:
            nn_module.train()
        
        for integrated_optimizer in self.integrated_optimizers:
            ModelMisc.train_or_eval_submodules(
                integrated_optimizer.root_module,
                integrated_optimizer.freeze_modules,
                False,
            )
        self.training = True
    
    def _eval_mode(self):
        # called in "before_validation"
        for nn_module in self.nn_module_list:
            nn_module.eval()
        self.training = False
    
    def _before_all_epochs(self, **kwargs):
        is_resumed = self._resume_training()
        if not is_resumed:
            self._load_pretrained_models()
        
        for integrated_optimizer in self.integrated_optimizers:
            ModelMisc.unfreeze_or_freeze_submodules(
                integrated_optimizer.root_module,
                integrated_optimizer.freeze_modules,
                False,
                )
            
            ModelMisc.unfreeze_or_freeze_params(
                integrated_optimizer.root_module,
                integrated_optimizer.freeze_params,
                False,
                )
            
        ModelMisc.show_model_info(self.cfg, self)
        self._get_pbar()
    
    def _before_one_epoch(self, **kwargs):
        self.epoch += 1
        if self.cfg.env.distributed:
            # shuffle data for each epoch (here needs epoch start from 0)
            self.train_loader.sampler_set_epoch(self.epoch - 1)  
        
        DistMisc.barrier()

        if DistMisc.is_main_process():
            if self.cfg.info.global_tqdm:
                self.train_pbar.unpause()
            else :
                self.train_pbar.reset()
                self.val_pbar.reset()
          
        self.step_count = 0
        for integrated_optimizer in self.integrated_optimizers:
            integrated_optimizer.zero_grad()
        self._train_mode()
    
    def _after_first_train_iter(self, **kwargs):
        pass
    
    def _after_one_epoch(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'train_epoch', self.train_outputs, self.trained_iters)
        
        self._save_checkpoint()
    
    def _before_validation(self, **kwargs):
        DistMisc.barrier()
        
        if DistMisc.is_main_process():
            self.val_pbar.unpause()
            
        self._eval_mode()
    
    def _after_first_validation_iter(self, **kwargs):
        pass
    
    def _after_validation(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'val_epoch', self.last_val_metrics, self.trained_iters)
        
        self._save_checkpoint_only_best_model()
    
    def _after_all_epochs(self, **kwargs):
        DistMisc.barrier()
        
        if DistMisc.is_main_process():
            self.train_pbar.close()
            self.val_pbar.close() 
    
    def _forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        inputs['train_progress'] = self.trained_iters / self.total_iters
        
        if self.training:
            outputs = self.ds_engine(inputs)
            loss_dict, metrics_dict = self.criterion(outputs, targets)
        else:
            with torch.no_grad():
                outputs = self.model_without_ddp(inputs)
                loss_dict, metrics_dict = self.criterion(outputs, targets)
                
                if self.ema_container is not None:
                    ema_outputs = self.ema_container(inputs)
                    ema_loss_dict, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                    outputs.update(LoggerMisc.set_dict_key_prefix(ema_outputs, 'ema_'))
                    loss_dict.update(ema_loss_dict)
                    metrics_dict.update(ema_metrics_dict)
            
        return outputs, loss_dict, metrics_dict
    
    def _backward_and_step(self, loss_dict: Dict[str, torch.Tensor]):
        def _compute_grad_norm():
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model_without_ddp.parameters(), max_norm=torch.inf)
            return grad_norm
        
        loss = loss_dict['loss_main']
        self.ds_engine.backward(loss)
        
        grad_norm_dict = {'grad_norm_main': _compute_grad_norm(self.ds_optimizer)}
        
        self.ds_engine.step()
        
        self.trained_iters += 1
        return grad_norm_dict
        
    def _train_one_epoch(self):
        cfg = self.cfg
        loggers = self.loggers
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=loggers,
            pbar=self.train_pbar,  
            header='Train',
            epoch_str=f'epoch: [{self.epoch}/{self.total_epochs}]',
            )
        mlogger.add_metrics([{
            **{f'loss_{integrated_optimizer.identifier}': ValueMetric(high_prior=True) for integrated_optimizer in self.integrated_optimizers},
            **{f'grad_norm_{integrated_optimizer.identifier}': ValueMetric(high_prior=True, no_sync=True) for integrated_optimizer in self.integrated_optimizers},
            **{lr_group: ValueMetric(format='{value:.2e}', final_format='[{min:.2e}, {max:.2e}]', low_prior=True, no_sync=True) for lr_group in self.lr_groups.keys()},
            'epoch': ValueMetric(window_size=1, no_print=True, no_sync=True),
            }])
        first_iter = True
        for batch in mlogger.log_every(self.train_loader):
            
            _, loss_dict, metrics_dict = self._forward(batch)
            
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                **self.lr_groups,
                **loss_dict,
                **metrics_dict,
            )
            
            grad_norm_dict = self._backward_and_step(loss_dict)
            
            mlogger.update_metrics(
                epoch=self.trained_iters / self.train_len,
            )
            for key, grad_norm in grad_norm_dict.items():
                if grad_norm is not None:
                    mlogger.update_metrics(**{key: grad_norm})
            
            if cfg.info.iter_log_freq > 0:
                if self.trained_iters % (cfg.info.iter_log_freq * cfg.trainer.grad_accumulation) == 0:
                    LoggerMisc.logging(loggers, 'train_iter', mlogger.output_dict(no_avg_list=['all']), self.trained_iters)
            
            if first_iter:
                first_iter = False
                self._after_first_train_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.forward_epoch_metrics())
        self.train_outputs = mlogger.output_dict(no_avg_list=[*self.lr_groups.keys(), 'epoch'], sync=True, final_print=True)
    
    def _evaluate(self):
        cfg = self.cfg
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=self.loggers,
            pbar=self.val_pbar,  
            header='Eval ',
            epoch_str=f'epoch: [{self.epoch}/{self.total_epochs}]',
            )
        mlogger.add_metrics([{f'loss_{integrated_optimizer.identifier}': ValueMetric(high_prior=True) for integrated_optimizer in self.integrated_optimizers}])
        first_iter = True
        for batch in mlogger.log_every(self.val_loader):
            
            _, loss_dict, metrics_dict = self._forward(batch)
            
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                **loss_dict,
                **metrics_dict,
            )
            
            if first_iter:
                first_iter = False
                self._after_first_validation_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.forward_epoch_metrics())
        if hasattr(self, 'ema_criterion'):
            mlogger.add_epoch_metrics(**self.ema_criterion.forward_epoch_metrics())
        self.last_val_metrics = mlogger.output_dict(sync=self.dist_eval, final_print=True)
        
    def _print_module_states(self, prefix):
        print(f'\n[{prefix}] epoch {self.epoch}:')
        DistMisc.avoid_print_mess()
        print(f'\tRank {DistMisc.get_rank()}:', force=True)
        print(f'\t\tOnline:', force=True)
        self.model_without_ddp.print_states(prefix='\t\t\t')
        self.criterion.print_states(prefix='\t\t\t')
        if self.ema_container is not None:
            print(f'\t\tEMA:', force=True)
            self.ema_container.ema_model.print_states(prefix='\t\t\t')
            self.ema_criterion.print_states(prefix='\t\t\t')
    
    def run(self):
        # train and val
        # prepare for 1. resumed training if needed; 2. show model information; 3. progress bar;
        self._before_all_epochs()
        
        for _ in self.epoch_loop:
            self._before_one_epoch()
            
            if self.cfg.info.print_module_states:
                self._print_module_states('Train')
            self._train_one_epoch()
            
            self._after_one_epoch()
            
            if self.epoch in self.val_epoch_list:
                
                self._before_validation()
                
                if self.dist_eval or DistMisc.is_main_process():
                    if self.cfg.info.print_module_states:
                        self._print_module_states('Eval')
                    self._evaluate()
                    
                self._after_validation()
                
                if self.cfg.special.debug == 'one_val_epoch':
                    PortalMisc.end_everything(self.cfg, self.loggers, force=True)
                
            if self.cfg.special.debug == 'one_epoch':
                PortalMisc.end_everything(self.cfg, self.loggers, force=True)
                
        self._after_all_epochs()
