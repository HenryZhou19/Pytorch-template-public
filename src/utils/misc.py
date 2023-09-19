import logging
import os
import pickle
import random
import signal
import sys
import time
import warnings
from argparse import Namespace
from glob import glob
from math import inf

import hjson
import numpy as np
import sacred
import torch
import torch.distributed as dist
import wandb
import yaml
from tqdm import tqdm


class ConfigMisc:
    @staticmethod
    def get_configs_from_sacred(main_config):
        ex = sacred.Experiment('Config Collector', save_git_info=False)
        ex.add_config(main_config)
        
        def print_sacred_configs(_run):
            final_config = _run.config
            final_config.pop('seed', None)
            config_mods = _run.config_modifications
            config_mods.pop('seed', None)
            print(sacred.commands._format_config(final_config, config_mods))

        @ex.main
        def print_init_config(_config, _run):
            if "RANK" in os.environ:
                rank = int(os.environ["RANK"])
                if rank != 0:
                    return
            print(f"\nInitial configs read by sacred for ALL Ranks:")
            print_sacred_configs(_run)

        config = ex.run_commandline().config
        cfg = ConfigMisc.nested_dict_to_nested_namespace(config)
        if hasattr(cfg, 'seed'):
            delattr(cfg, 'seed')  # seed given by sacred is useless
        
        return cfg

    @staticmethod 
    def nested_dict_to_nested_namespace(dictionary):
        namespace = dictionary
        if isinstance(dictionary, dict):
            namespace = Namespace(**dictionary)
            for key, value in dictionary.items():
                setattr(namespace, key, ConfigMisc.nested_dict_to_nested_namespace(value))
        return namespace
    
    @staticmethod 
    def nested_namespace_to_nested_dict(namespace):
        dictionary = {}
        for name, value in vars(namespace).items():
            if isinstance(value, Namespace):
                dictionary[name] = ConfigMisc.nested_namespace_to_nested_dict(value)
            else:
                dictionary[name] = value
        return dictionary
    
    @staticmethod
    def nested_namespace_to_plain_namespace(namespace):
        def setattr_safely(ns, n, v):
            assert not hasattr(ns, n), f'Namespace conflict: {v}'
            setattr(ns, n, v)
        
        plain_namespace = Namespace()
        for name, value in vars(namespace).items():
            if isinstance(value, Namespace):
                plain_subnamespace = ConfigMisc.nested_namespace_to_plain_namespace(value)
                for subname, subvalue in vars(plain_subnamespace).items():
                    setattr_safely(plain_namespace, subname, subvalue)
            else:
                setattr_safely(plain_namespace, name, value)
        
        return plain_namespace
    
    @staticmethod
    def update_nested_namespace(cfg_base, cfg_new):
        for name, value in vars(cfg_new).items():
            if isinstance(value, Namespace):
                if name not in vars(cfg_base):
                    setattr(cfg_base, name, Namespace())
                ConfigMisc.update_nested_namespace(getattr(cfg_base, name), value)
            else:
                setattr(cfg_base, name, value)

    @staticmethod
    def read(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        return ConfigMisc.nested_dict_to_nested_namespace(config)

    @staticmethod
    def write(path, config):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(ConfigMisc.nested_namespace_to_nested_dict(config), f)

    @staticmethod
    def get_specific_list(cfg, cfg_keys):
        def get_nested_attr(cfg, key):
            if '.' in key:
                key, subkey = key.split('.', 1)
                return get_nested_attr(getattr(cfg, key), subkey)
            else:
                return getattr(cfg, key)
        return [str(get_nested_attr(cfg, extra)) for extra in cfg_keys]

    @staticmethod
    def output_dir_extras(cfg):
        extras = '_'.join([cfg.info.start_time] + ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
        if cfg.special.debug:
            extras = 'debug_' + extras
        return extras


class PortalMisc:
    @staticmethod
    def combine_train_infer_configs(infer_cfg, use_train_seed=True):
        cfg = ConfigMisc.read(infer_cfg.tester.train_cfg_path)
        train_seed = cfg.env.seed
        ConfigMisc.update_nested_namespace(cfg, infer_cfg)
        if use_train_seed:
            cfg.env.seed = train_seed

        cfg.env.distributed = False
        cfg.info.train_work_dir = cfg.info.work_dir
        cfg.info.work_dir = cfg.info.train_work_dir + '/inference_results/' + cfg.info.infer_start_time
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
        checkpoint_path = glob(os.path.join(
            cfg.info.train_work_dir,
            'checkpoint_best_epoch_*.pth' if cfg.tester.use_best else 'checkpoint_last_epoch_*.pth'))
        assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
        cfg.tester.checkpoint_path = checkpoint_path[0]

        return cfg

    @staticmethod 
    def resume_or_new_train_dir(cfg):  # only for train
        assert hasattr(cfg.env, 'distributed')
        if cfg.trainer.resume is not None:  # read "work_dir", "start_time" from the .yaml file
            print('Resuming from: ', cfg.trainer.resume, ', reading configs from .yaml file...')
            cfg_old = ConfigMisc.read(cfg.trainer.resume)
            work_dir = cfg_old.info.work_dir
            setattr(cfg.info, 'resume_start_time', cfg.info.start_time)
            cfg.info.start_time = cfg_old.info.start_time
        else:
            work_dir = cfg.info.output_dir + ConfigMisc.output_dir_extras(cfg)
            if DistMisc.is_main_process():
                print('New start at: ', work_dir)
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
            cfg.info.work_dir = work_dir
        cfg.info.work_dir = work_dir

    @staticmethod
    def seed_everything(cfg):
        assert hasattr(cfg.env, 'distributed')
        if cfg.env.seed_with_rank:
            cfg.env.seed = cfg.env.seed + DistMisc.get_rank()
        
        os.environ['PYTHONHASHSEED'] = str(cfg.env.seed)

        random.seed(cfg.env.seed)
        np.random.seed(cfg.env.seed)
        
        torch.manual_seed(cfg.env.seed)
        torch.cuda.manual_seed(cfg.env.seed)
        torch.cuda.manual_seed_all(cfg.env.seed)

        if cfg.env.cuda_deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    @staticmethod
    def special_config_adjustment(cfg):
        if cfg.special.debug:  # debug mode
            cfg.env.num_workers = 0

    @staticmethod
    def save_configs(cfg):
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
            if cfg.trainer.resume==None:
                cfg_file_name = 'cfg.yaml'
            else:
                cfg_file_name = f'cfg_resume_{cfg.info.resume_start_time}.yaml'
            ConfigMisc.write(os.path.join(cfg.info.work_dir, cfg_file_name), cfg)

    @staticmethod
    def force_print_config(cfg, force=False):
        def str_block_wrapper(str, block_width=80):
            return '\n' + '='*block_width + '\n' + str + '='*block_width + '\n'
        
        def write_msg_lines(msg_in, cfg_in, indent=1):
            for m in sorted(vars(cfg_in).keys()):
                m_indent = ' ' * (4*(indent - 1)) + ' ├─ ' + m
                v = cfg_in.__getattribute__(m)
                if isinstance(v, Namespace):
                    msg_in += write_msg_lines(f'{m_indent}\n', v, indent + 1)
                else:
                    if len(m_indent) > 40:
                        warnings.warn(f'Config key "{m}" with indent is too long (>40) to display, please check.')
                    if len(m_indent) < 38:
                        m_indent += ' ' + '-' * (38 - len(m_indent)) + ' '
                    msg_in += f'{m_indent:40}{v}\n'
            return msg_in

        msg = f"Rank {DistMisc.get_rank()} --- Parameters:\n"
        msg = StrMisc.block_wrapper(write_msg_lines(msg, cfg), s='=', block_width=80)

        DistMisc.avoid_print_mess()
        if cfg.env.distributed:
            print(msg, force=True)
        else:
            print(msg)
        DistMisc.avoid_print_mess()

    @staticmethod 
    def init_loggers(cfg):
        if DistMisc.is_main_process():
            wandb_name = '_'.join(ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
            wandb_name = f'[{cfg.info.task_type}] ' + wandb_name
            wandb_tags = ConfigMisc.get_specific_list(cfg, cfg.info.wandb_tags)
            if TesterMisc.for_inference(cfg):
                wandb_tags.append(f'Infer: {cfg.info.infer_start_time}')
            if cfg.trainer.resume != None:
                wandb_tags.append(f'Re: {cfg.info.resume_start_time}')
            cfg.info.wandb_run = wandb.init(
                project=cfg.info.project_name,
                name=wandb_name,
                tags=wandb_tags,
                dir=cfg.info.work_dir,
                config=ConfigMisc.nested_namespace_to_plain_namespace(cfg)
                )
            cfg.info.log_file = open(os.path.join(cfg.info.work_dir, 'logs.txt'), 'a' if cfg.trainer.resume is None else 'a+')
        else:
            cfg.info.log_file = sys.stdout

    @staticmethod 
    def end_everything(cfg, end_with_printed_cfg=False, force=False):
        if end_with_printed_cfg:
            PortalMisc.force_print_config(cfg)
        try:
            if DistMisc.is_main_process():
                cfg.info.log_file.close()
                print('log_file closed.')
                if force:
                    wandb.finish(exit_code=-1)              
                else:
                    wandb.finish()
                print('wandb closed.')
        finally:
            exit()           


    @staticmethod 
    def interrupt_handler(cfg):
        """Handles SIGINT signal (Ctrl+C) by exiting the program gracefully."""
        def signal_handler(sig, frame):
            print('Received SIGINT. Cleaning up...')
            PortalMisc.end_everything(cfg, force=True)

        signal.signal(signal.SIGINT, signal_handler)


class DistMisc:
    @staticmethod
    def avoid_print_mess():
        if DistMisc.is_dist_avail_and_initialized():  # 
            dist.barrier()
            time.sleep(DistMisc.get_rank() * 0.1)
    
    @staticmethod
    def all_gather(data):

        """
        Run all_gather on arbitrary picklable data (not necessarily tensors)
        Args:
            data: any picklable object
        Returns:
            list[data]: list of data gathered from each rank
        """
        world_size = DistMisc.get_world_size()
        if world_size == 1:
            return [data]

        # serialized to a Tensor
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # obtain Tensor size of each rank
        local_size = torch.tensor([tensor.numel()], device="cuda")
        size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        if local_size != max_size:
            padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        dist.all_gather(tensor_list, tensor)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list

    @staticmethod
    def reduce_dict(input_dict, average=True):
        world_size = DistMisc.get_world_size()
        if world_size < 2:
            return input_dict
        with torch.inference_mode():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.all_reduce(values)
            if average:
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    @staticmethod
    def reduce_sum(tensor):
        world_size = DistMisc.get_world_size()
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def reduce_mean(tensor):
        world_size = DistMisc.get_world_size()
        total = DistMisc.reduce_sum(tensor)
        return total.float() / world_size

    @ staticmethod
    def is_dist_avail_and_initialized():
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def get_world_size():
        return dist.get_world_size() if DistMisc.is_dist_avail_and_initialized() else 1

    @staticmethod
    def get_rank():
        return dist.get_rank() if DistMisc.is_dist_avail_and_initialized() else 0

    @staticmethod
    def is_main_process():
        return DistMisc.get_rank() == 0

    @staticmethod
    def setup_for_distributed(is_master):
        # This function disables printing when not in master process
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def dist_print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = dist_print

    @staticmethod
    def init_distributed_mode(cfg):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            cfg.env.rank = int(os.environ["RANK"])
            cfg.env.world_size = int(os.environ["WORLD_SIZE"])
            cfg.env.gpu = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ and 'SLURM_PTY_PORT' not in os.environ:
            cfg.env.rank = int(os.environ["SLURM_PROCID"])
            cfg.env.gpu = cfg.env.rank % torch.cuda.device_count()
        else:
            print("Not using distributed mode")
            cfg.env.distributed = False
            cfg.data.batch_size_total = cfg.data.batch_size_per_rank
            return

        cfg.env.distributed = True
        cfg.env.dist_backend = 'nccl'
        cfg.data.batch_size_total = cfg.data.batch_size_per_rank * cfg.env.world_size
        torch.cuda.set_device(cfg.env.gpu)
        
        dist.distributed_c10d.logger.setLevel(logging.WARNING)
        
        dist.init_process_group(
            backend=cfg.env.dist_backend, init_method=cfg.env.dist_url, world_size=cfg.env.world_size, rank=cfg.env.rank
        )       
        # DistMisc.avoid_print_mess()
        # print(f"INFO - distributed init (Rank {cfg.env.rank}): {cfg.env.dist_url}")
        # DistMisc.avoid_print_mess()
        DistMisc.setup_for_distributed(cfg.env.rank == 0)


class ModelMisc:
    @staticmethod
    def print_trainable_params(model):
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    @staticmethod
    def ddp_wrapper(cfg, model_without_ddp):
        model = model_without_ddp
        if cfg.env.distributed:
            if cfg.env.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.env.gpu],
                find_unused_parameters=cfg.env.find_unused_params,
            )
        return model
    
    @staticmethod
    def deepspeed_ddp_wrapper(cfg, model_without_ddp):
        print(StrMisc.block_wrapper('Using DeepSpeed DDP wrapper...\n', s='#', block_width=80))
        DistMisc.avoid_print_mess()
        import deepspeed
        deepspeed.logger.setLevel(logging.WARNING)
        def ds_init_engine_wrapper() -> deepspeed.DeepSpeedEngine:
            return deepspeed.initialize(model=model_without_ddp, config=deepspeed_config)[0]
        with open(cfg.deepspeed.deepspeed_config, 'r') as json_file:
            deepspeed_config = hjson.load(json_file)
        deepspeed_config.update({'train_batch_size': cfg.data.batch_size_total})
        return ds_init_engine_wrapper()


class OptimizerMisc:
    @staticmethod
    def get_param_dicts_with_specific_lr(cfg, model_without_ddp: torch.nn.Module):
        def match_name_keywords(name, name_keywords):
            for keyword in name_keywords:
                if keyword in name:
                    return True
            return False

        if not hasattr(cfg.trainer, 'lr'):
            assert hasattr(cfg.trainer, 'lr_groups')
            param_dicts_with_lr = [
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, 'backbone') and p.requires_grad],
                "lr": cfg.trainer.lr_groups.main,},
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if match_name_keywords(n, 'backbone') and p.requires_grad],
                "lr": cfg.trainer.lr_groups.backbone},
                ]
        else:  # if cfg.trainer.lr exists, then all params use cfg.trainer.lr
            param_dicts_with_lr = [
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if p.requires_grad],
                "lr": cfg.trainer.lr},
                ]
        
        return param_dicts_with_lr
    

class TrainerMisc:
    @staticmethod
    def get_pbar(cfg, trainer_status):
        if DistMisc.is_main_process():
            len_train_loader = len(trainer_status['train_loader'])
            len_val_loader = len(trainer_status['val_loader'])
            epoch_finished = trainer_status['start_epoch'] - 1
            train_pbar = tqdm(
                total=cfg.trainer.epochs*len_train_loader if cfg.info.global_tqdm else len_train_loader,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
                initial=epoch_finished*len_train_loader,
            )
            train_pbar.set_description_str('Train')
            print('')
            val_pbar = tqdm(
                total=cfg.trainer.epochs*len_val_loader if cfg.info.global_tqdm else len_val_loader,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
                initial=epoch_finished*len_val_loader,
            )
            val_pbar.set_description_str('Eval ')
            print('')
            trainer_status['train_pbar'] = train_pbar
            trainer_status['val_pbar'] = val_pbar
            
        return trainer_status
    
    @staticmethod
    def resume_training(cfg, trainer_status):
        if cfg.trainer.resume:
            print('Resuming from ', cfg.trainer.resume, ', loading the checkpoint...')
            checkpoint_path = glob(os.path.join(cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
            assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
            checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
            trainer_status['model_without_ddp'].load_state_dict(checkpoint['model'])
            trainer_status['optimizer'].load_state_dict(checkpoint['optimizer'])
            trainer_status['lr_scheduler'].load_state_dict(checkpoint['lr_scheduler'])
            if cfg.env.amp:
                trainer_status['scaler'].load_state_dict(checkpoint['scaler'])
            trainer_status['start_epoch'] = checkpoint['epoch'] + 1
            trainer_status['best_metrics'] = checkpoint.get('best_metrics', {})
            trainer_status['metrics'] = checkpoint.get('last_metrics', {})
            trainer_status['train_iters'] = checkpoint['epoch']*len(trainer_status['train_loader'])
        else:
            print('New trainer.')
        print(f"Start from epoch: {trainer_status['start_epoch']}")
        
        return trainer_status
    
    @staticmethod
    def before_one_epoch(cfg, trainer_status, **kwargs):
        assert 'epoch' in kwargs.keys()
        trainer_status['epoch'] = kwargs['epoch']
        if cfg.env.distributed:
            # shuffle data for each epoch (here needs epoch start from 0)
            trainer_status['train_loader'].sampler_set_epoch(trainer_status['epoch'] - 1)  
        
        if DistMisc.is_main_process():
            if cfg.info.global_tqdm:
                trainer_status['train_pbar'].unpause()
            else :
                trainer_status['train_pbar'].reset()
                trainer_status['val_pbar'].reset()

    @staticmethod
    def after_training_before_validation(cfg, trainer_status, **kwargs):
        TrainerMisc.wandb_log(cfg,  'train_', trainer_status['train_outputs'], trainer_status['train_iters'])

        if DistMisc.is_main_process():        
            trainer_status['val_pbar'].unpause()

    @staticmethod
    def after_validation(cfg, trainer_status, **kwargs):
        TrainerMisc.wandb_log(cfg, 'val_', trainer_status['metrics'], trainer_status['train_iters'])
        
        TrainerMisc.save_checkpoint(cfg, trainer_status)
    
    @staticmethod
    def after_all_epochs(cfg, trainer_status, **kwargs):
        if DistMisc.is_main_process():
            trainer_status['train_pbar'].close()
            trainer_status['val_pbar'].close()
    
    @staticmethod   
    def wandb_log(cfg, prefix, output_dict, step): 
        if DistMisc.is_main_process():          
            for k, v in output_dict.items():
                if k == 'epoch':
                    wandb.log({f'{k}': v}, step=step)  # log epoch without prefix
                else:
                    wandb.log({f'{prefix}{k}': v}, step=step)
                # wandb.log({'output_image': [wandb.Image(trainer_status['output_image'])]})
                # wandb.log({"output_video": wandb.Video(trainer_status['output_video'], fps=30, format="mp4")})
            
    @staticmethod
    def save_checkpoint(cfg, trainer_status):
        if DistMisc.is_main_process():
            epoch_finished = trainer_status['epoch']
            trainer_status['best_metrics'], save_flag = trainer_status['metric_criterion'].choose_best(
                trainer_status['metrics'], trainer_status['best_metrics']
            )


            save_files = {
                'model': trainer_status['model_without_ddp'].state_dict(),
                'best_metrics': trainer_status['best_metrics'],
                'epoch': epoch_finished,
            }

            if save_flag:
                best = glob(os.path.join(cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
                assert len(best) <= 1
                if len(best) == 1:
                    torch.save(save_files, best[0])
                    os.rename(best[0], os.path.join(cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))

            if (trainer_status['epoch'] + 1) % cfg.trainer.save_interval == 0:
                save_files.update({
                    'optimizer': trainer_status['optimizer'].state_dict(),
                    'lr_scheduler': trainer_status['lr_scheduler'].state_dict(),
                    'last_metric': trainer_status['metrics']
                })
                if cfg.env.amp and cfg.env.device:
                    save_files.update({
                        'scaler': trainer_status['scaler'].state_dict()
                    })
                last = glob(os.path.join(cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
                assert len(last) <= 1
                if len(last) == 1:
                    torch.save(save_files, last[0])
                    os.rename(last[0], os.path.join(cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))


class TesterMisc:
    @staticmethod
    def get_pbar(cfg, tester_status):
        if DistMisc.is_main_process():
            test_pbar = tqdm(
                total=len(tester_status['test_loader']),
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
            )
            test_pbar.set_description_str('Test ')
            print('')
            tester_status['test_pbar'] = test_pbar
        
        return tester_status
    
    @staticmethod
    def for_inference(cfg):
        return hasattr(cfg, 'tester')
    
    @staticmethod
    def load_model(cfg, tester_status):
        checkpoint = torch.load(cfg.tester.checkpoint_path, map_location=tester_status['device'])
        tester_status['model_without_ddp'].load_state_dict(checkpoint['model'])
        # print(f'{config.mode} mode: Loading pth from', path)
        print('Loading pth from', cfg.tester.checkpoint_path)
        print('best_trainer_metric', checkpoint.get('best_metric', {}))
        if DistMisc.is_main_process():
            if 'epoch' in checkpoint.keys():
                print('Epoch:', checkpoint['epoch'])
                cfg.info.wandb_run.tags = cfg.info.wandb_run.tags + (f"Epoch: {checkpoint['epoch']}",)
        print('last_trainer_metric', checkpoint.get('last_metric', {}))
        
        return tester_status

    @staticmethod
    def before_inference(cfg, tester_status, **kwargs):
        pass

    @staticmethod
    def after_inference(cfg, tester_status, **kwargs):
        if DistMisc.is_main_process():
            for k, v in tester_status['metrics'].items():
                wandb.log({f'infer_{k}': v})
                # wandb.log({'output_image': [wandb.Image(tester_status['output_image'])]})
                # wandb.log({"output_video": wandb.Video(tester_status['output_video'], fps=30, format="mp4")})
            
            tester_status['test_pbar'].close()


class StrMisc:
    @staticmethod
    def block_wrapper(str, s='=', block_width=80):
        return '\n' + s*block_width + '\n' + str + s*block_width + '\n'

class TimeMisc:
    @staticmethod
    def get_time_str():
        return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
 
    class Timer:
        def __init__(self):
            self.t_start= time.time()
            self.t = self.t_start
            
        def press(self):
            self.t = time.time()

        def restart(self):
            self.__init__()

        @property
        def info(self):
            now = time.time()
            return {
                'all': now - self.t_start,
                'last': now - self.t
                }
        
    class TimerContext:
        def __init__(self, block_name, do_print=True):
            self.block_name = block_name
            self.do_print = do_print
            
        def __enter__(self):
            self.timer = TimeMisc.Timer()

        def __exit__(self, *_):
            if self.do_print:
                m_indent = '    ' + self.block_name
                if len(m_indent) > 40:
                    warnings.warn(f'Block name "{self.block_name}" with indent is too long (>40) to display, please check.')
                if len(m_indent) < 38:
                        m_indent += ' ' + '-' * (38 - len(m_indent)) + ' '
                print(f"{m_indent:40s}elapsed time: {self.timer.info['all']:.4f}")
