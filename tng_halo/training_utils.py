
import torch
import torch.nn as nn
import math

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))))

def get_activation(activation_name):
    """ Get an activation function. """
    if activation_name.lower() == 'identity':
        return nn.Identity()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name.lower() == 'leaky_relu':
        alpha = activation.get('leaky_relu_alpha', 0.01)
        return nn.LeakyReLU(alpha)
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {activation_name}')

def configure_optimizers(parameters, optimizer_args, scheduler_args=None):
    """ Return optimizer and scheduler for Pytorch Lightning """
    scheduler_args = scheduler_args or {}

    # setup the optimizer
    if optimizer_args.name == "Adam":
        optimizer = torch.optim.Adam(
            parameters, lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    elif optimizer_args.name == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters, lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    else:
        raise NotImplementedError(
            "Optimizer {} not implemented".format(optimizer_args.name))

    # setup the scheduler
    if scheduler_args.name is None:
        scheduler = None
    elif scheduler_args.name == 'ReduceLROnPlateau':
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=scheduler_args.factor,
            patience=scheduler_args.patience)
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min)
    elif scheduler_args.name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args.decay_steps,
            warmup_steps=scheduler_args.warmup_steps,
            eta_min=scheduler_args.eta_min)
    else:
        raise NotImplementedError(
            "Scheduler {} not implemented".format(self.scheduler_args.name))

    if scheduler is None:
        return optimizer
    else:
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': scheduler_args.interval,
                'frequency': 1
            }
        }
