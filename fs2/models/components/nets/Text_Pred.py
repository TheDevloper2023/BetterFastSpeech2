import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            DPReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)

    
class StyleAdaptor(nn.Module):
    def __init__(self, dropout=0.1, in_dim=768, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            DPReLU(),
            nn.Dropout(dropout),

            ResBlock(512, dropout),
            ResBlock(512, dropout),

            nn.Linear(512, 256),
            DPReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)





# from https://github.com/ZDisket/FastSpeech2
class DPReLU(nn.Module):
    """
    DPReLU: A dynamic ReLU variant:

    "There are four additional learnable parameters compared to the vanilla ReLU. alpha and beta are the slopes of the negative
    and positive parts in the function, respectively. Here, a negative or positive case is determined when comparing input
    x to the threshold. The threshold makes DPReLU shift on the x-axis in comparison to the original ReLU. The bias
    determines the alignment of the function with respect to the y-axis. These four parameters are all learnable and interact
    with each other during the training phase"

    https://link.springer.com/article/10.1007/s44196-023-00186-w

    By default, alpha and beta are 0.5 and 0.9, which yielded best results according to the paper. Threshold and bias = 0.

    Important: Please use He or their custom initialization!

    Converted from Tensorflow from https://github.com/KienMN/Activation-Experiments/tree/master
    """

    def __init__(self, alpha_init=0.5, beta_init=0.9, threshold_init=0.0, bias_init=0.0, shared_axes=None):
        super(DPReLU, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

        self.shared_axes = shared_axes
        if self.shared_axes is not None and not isinstance(self.shared_axes, (list, tuple)):
            self.shared_axes = [self.shared_axes]

    def forward(self, inputs):
        neg = -self.alpha * torch.relu(-inputs + self.threshold)
        pos = self.beta * torch.relu(inputs - self.threshold)
        return pos + neg + self.bias

    def extra_repr(self):
        return f'alpha={self.alpha.item()}, beta={self.beta.item()}, threshold={self.threshold.item()}, bias={self.bias.item()}, shared_axes={self.shared_axes}'