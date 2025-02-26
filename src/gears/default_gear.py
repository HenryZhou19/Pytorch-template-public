import torch

from .modules.deepspeed_trainer_base import DeepspeedTrainerBase
from .modules.tester_base import TesterBase, tester_register
from .modules.trainer_base import TrainerBase, trainer_register


@trainer_register('default')
class Trainer(TrainerBase):
    def _train_mode(self):
        super()._train_mode()
    
    def _eval_mode(self):
        super()._eval_mode()
    
    def _before_all_epochs(self, **kwargs):
        super()._before_all_epochs(**kwargs)
    
    def _before_one_epoch(self, **kwargs):
        super()._before_one_epoch(**kwargs)
    
    def _after_first_train_iter(self, **kwargs):
        super()._after_first_train_iter(**kwargs)
    
    def _after_one_epoch(self, **kwargs):
        super()._after_one_epoch(**kwargs)
    
    def _before_validation(self, **kwargs):
        super()._before_validation(**kwargs)
    
    def _after_first_validation_iter(self, **kwargs):
        super()._after_first_validation_iter(**kwargs)
    
    def _after_validation(self, **kwargs):
        super()._after_validation(**kwargs)
    
    def _after_all_epochs(self, **kwargs):
        super()._after_all_epochs(**kwargs)
    
    def _forward(self, batch: dict):
        return super()._forward(batch)
    
    def _backward_and_step(self, loss: torch.Tensor):
        return super()._backward_and_step(loss)
    
    def _train_one_epoch(self):
        super()._train_one_epoch()
    
    def _evaluate(self):
        super()._evaluate()
        

@trainer_register('deepspeed')
class DeepspeedTrainer(DeepspeedTrainerBase):
    pass


@tester_register('default')
class Tester(TesterBase):
    def _eval_mode(self):
        super()._eval_mode()
    
    def _before_inference(self, **kwargs):
        super()._before_inference(**kwargs)
    
    def _after_first_inference_iter(self, **kwargs):
        super()._after_first_inference_iter(**kwargs)
    
    def _after_inference(self, **kwargs):
        super()._after_inference(**kwargs)
    
    def _forward(self, batch: dict):
        return super()._forward(batch)
    
    def _test(self):
        super()._test()
    