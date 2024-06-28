import torch
import torch.nn as nn
from utils import init_max_weights
from LMF_fusion import LMF

class NeuralNet(nn.Module):
 def __init__(self, input_size, hidden_size, num_classes):
     super(NeuralNet, self).__init__()
     self.fc1 = nn.Linear(input_size, hidden_size) #输入层　 #隐藏网络：elu的功能是将输入的feature的tensor所有的元素中如果小于零的就取零。
     self.relu = nn.ReLU()
     self.fc2 = nn.Linear(hidden_size, num_classes) #输出层
     self.sigmoid = nn.Sigmoid()

 def forward(self, x):
     out = self.fc1(x)
     out = self.relu(out)
     out = self.fc2(out)
     out = self.sigmoid(out)
     return out
#
# class CBAMLayer(nn.Module):
#     def __init__(self, channel, reduction=1, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x

class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder3 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 33), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

        self.NeuralNet = NeuralNet(130,75,130)

        # lmf
        self.lmf = LMF()


    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)

        o12 = self.lmf(o1, o2).flatten(start_dim=1)

        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            weight = self.NeuralNet((torch.cat((out,o1,o2),1)))
            out = torch.cat((out*weight[:,0:64], o1*weight[:,64:97], o2*weight[:,97:]), 1)
        out = self.encoder2(out)

        return out
