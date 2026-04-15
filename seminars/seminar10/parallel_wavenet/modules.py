from math import sqrt

import torch
from torch import nn


"""
Introduce auxiliary modules:
1. causal convolution – simple convolution with `kernel_size` and `dilation` 
hyper-parameters, but working in causal way (does not look in the future)
2. residual block – main building component of WaveNet architecture
"""

class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv, self).__init__()

        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation, cin_channels):
        super(ResBlock, self).__init__()
        self.cin_channels = cin_channels

        self.filter_conv = CausalConv(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv(in_channels, out_channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
        self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
        self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
        self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
        nn.init.kaiming_normal_(self.filter_conv_c.weight)
        nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, x, c=None):
        h_filter = self.filter_conv(x)
        h_gate = self.gate_conv(x)
        h_filter += self.filter_conv_c(c)
        h_gate += self.gate_conv_c(c)
        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * sqrt(0.5), skip
