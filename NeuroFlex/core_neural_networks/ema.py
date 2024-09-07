import torch
from torch import nn

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        self.module = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.shadow = {}
        self.num_updates = 0

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name])

    def store(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def restore(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def __call__(self, model):
        self.update()
        return self.module

    def buffers(self):
        return self.shadow.values()
