
import warnings
import torch

from torch import nn

warnings.filterwarnings('ignore')


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # inter_channels = int(in_planes // r)
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_planes, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, out_planes, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(out_planes),
        # )

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MS_CAM(nn.Module):
    def __init__(self, in_channels, out_channels, r=4):
        super(MS_CAM, self).__init__()

        inter_channels = int(in_channels // r)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.global_att(x)
        return  weight

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class FBA(nn.Module):

    def __init__(self, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, 1)

        self.conv = nn.Sequential(BasicConv2d(input_dim, input_dim, 3, 1, 1),
                                  nn.Conv2d(input_dim, 768, kernel_size=1, bias=False))
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

        self.stream1 = nn.Sequential(
            # nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=1),
            nn.Linear(576,1369),

        )
        self.lr = nn.Sequential(
            nn.LayerNorm(input_dim // 2),
            nn.ReLU()
        )
        self.lr2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )
        self.stream = nn.Sequential(
            # nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=1),
            nn.Linear(576,1369),
            # nn.LayerNorm(input_dim),
            # nn.ReLU()
        )
        self.stream2 = nn.Sequential(
            # nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=1),
            nn.Linear(576, 1369),
            # nn.LayerNorm(input_dim // 2),
            # nn.ReLU()
        )

    def forward(self, H_feature, L_feature):
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])

        out = self.conv(torch.cat([H_feature, L_feature], dim=1))

        split_size = out.shape[1] // 2
        # out = self.trans_conv(out)
        x1 = out[:, :split_size]
        x2 = out[:, split_size:]
        b,c=x1.shape[0],x1.shape[1]
        out=out.view(b, c * 2, -1)
        out = self.stream(out)
        out = self.lr2(out.permute(0,2,1)).permute(0,2,1)
        out = out.view(b,c*2,37,37)
        x1 = x1.view(b, c, -1)
        x2 = x2.view(b, c, -1)
        x1 = self.stream1(x1)

        x2 = self.stream2(x2)
        x1 = self.lr(x1.permute(0,2,1)).permute(0,2,1).view(b,c,37,37)
        x2 = self.lr(x2.permute(0,2,1)).permute(0,2,1).view(b,c,37,37)

        x = torch.cat((x1, x2), dim=1)
        out2 = self.Sigmoid(x)
        out = out * out2
        return out

