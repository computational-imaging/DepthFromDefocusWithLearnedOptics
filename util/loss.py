import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.metrics import Metric


class Vgg16PerceptualLoss(Metric):

    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg_blocks = nn.ModuleList([
            vgg16.features[:4].eval(),
            vgg16.features[4:9].eval(),
            vgg16.features[9:16].eval(),
        ])
        self.vgg_blocks.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.weight = [11.17 / 35.04 / 4, 35.04 / 35.04 / 4, 29.09 / 35.04 / 4]

        self.add_state('diff', default=torch.tensor([0., 0., 0., 0.]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0, 0, 0., 0.]), dist_reduce_fx='sum')

    def train_loss(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = F.l1_loss(input, target) / 4
        input = F.pad(input, mode='reflect', pad=(4, 4, 4, 4))
        target = F.pad(target, mode='reflect', pad=(4, 4, 4, 4))
        for i, block in enumerate(self.vgg_blocks):
            input = block(input)
            target = block(target)
            loss += self.weight[i] * F.l1_loss(input[..., 4:-4, 4:-4], target[..., 4:-4, 4:-4])
        return loss

    def update(self, input: torch.Tensor, target: torch.Tensor):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        self.diff[0] += (input - target).sum() / 4
        self.total[0] += input.numel()

        input = F.pad(input, mode='reflect', pad=(4, 4, 4, 4))
        target = F.pad(target, mode='reflect', pad=(4, 4, 4, 4))
        for i, block in enumerate(self.vgg_blocks):
            input = block(input)
            target = block(target)
            self.diff[i + 1] += self.weight[i] * (input[..., 4:-4, 4:-4] - target[..., 4:-4, 4:-4]).sum()
            self.total[i + 1] += input[..., 4:-4, 4:-4].numel()

    def compute(self) -> torch.Tensor:
        return self.diff[0] / self.total[0] + \
               self.diff[1] / self.total[1] + \
               self.diff[2] / self.total[2] + \
               self.diff[3] / self.total[3]
