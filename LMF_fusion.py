import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import random
class LMF(nn.Module):
    def __init__(self, rank=1, hidden_dims=[31, 31],output_dim=33):
        super(LMF, self).__init__()
        self.rank = rank
        self.audio_hidden = hidden_dims[0]
        self.output_dim = output_dim
        self.video_hidden = hidden_dims[1]


        self.audio_factor = Parameter(torch.randn(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.randn(self.rank, self.video_hidden + 1, self.output_dim))

    def forward(self, o1, o2):
        x1 = torch.unsqueeze(o1,dim=2)
        x2 = torch.transpose(self.audio_factor.cuda(),1,0)
        # print('x1_shape:', x1.shape)
        # print('x2_shape:', x2.shape)
        if x1.shape[0] != 32:
            fusion_o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)  # BATCH_SIZE X 1024
        else:
            o1 = torch.matmul(x1,x2) #32, 1, 33
            o2 = torch.matmul(torch.unsqueeze(o2,dim=2), torch.transpose(self.video_factor.cuda(),1,0))
            fusion_o12 = o1 * o2

        # o2 = torch.matmul(o2, self.video_factor.cuda())

        return fusion_o12.cuda()