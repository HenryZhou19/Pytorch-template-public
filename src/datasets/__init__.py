from torch.utils.data import (Dataset, RandomSampler, Sampler,
                              SequentialSampler, distributed)

from src.utils.misc import ImportMisc

from .modules.data_module_base import DataLoaderX, DataModuleBase
from .modules.data_module_register import get_data_module

ImportMisc.import_current_dir_all(__file__, __name__)

class DataManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_module = self._get_data_module()
        
    def _get_data_module(self) -> DataModuleBase:  # DataModule provides methods for getting train/val/test datasets
        data_module = get_data_module(self.cfg.data.dataset)(self.cfg)
        return data_module
        
    def build_dataset(self, split=None, shuffle=False) -> DataLoaderX:
        assert split in ['train', 'val', 'test'], f'Invalid split {split}'
        if split == 'train':
            dataset = self.data_module.get_train_dataset()
        elif split == 'val':
            dataset = self.data_module.get_val_dataset()
        else: # split == 'test':
            dataset = self.data_module.get_test_dataset()
        
        dist_sampler = True if split == 'train' else self.cfg.trainer.dist_eval
        dataloader = self._get_dataloader(dataset, shuffle=shuffle, dist_sampler=dist_sampler)
        print(f'{split} dataloader built successfully.')
        return dataloader
        
    def _get_dataloader(self, dataset: Dataset, shuffle: bool, dist_sampler: bool) -> DataLoaderX:
        sampler = self._get_sampler(dataset, shuffle, dist_sampler)

        return DataLoaderX(
            dataset,
            self.cfg.data.batch_size_per_rank,
            sampler=sampler,
            pin_memory=self.cfg.env.pin_memory,
            collate_fn=self.data_module.collate_fn,
            num_workers=self.cfg.env.num_workers,
            worker_init_fn=self.data_module.get_worker_init_fn(),
            generator=self.data_module.get_generator(),
            persistent_workers=True if self.cfg.env.num_workers > 0 else False,
        )
        
    def _get_sampler(self, dataset: Dataset, shuffle: bool, dist_sampler: bool) -> Sampler:
        if self.cfg.env.distributed and dist_sampler:
            sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler
