import torch
from torch import nn

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):

        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size-1)*dilation

        # modules:
        self.conv1d = nn.utils.weight_norm(torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=stride,
                                      padding=(kernel_size-1)*dilation,
                                      dilation=dilation), name = 'weight')

    def forward(self, seq):

        conv1d_out = self.conv1d(seq)
        # remove k-1 values from the end:
        return conv1d_out[:,:,:-(self.padding)]
