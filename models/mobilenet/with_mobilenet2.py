import torch
from torch import nn
from modules.conv import *

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)  # 입력: (batch_size, 512, H/8, W/8)   # align 후: (batch_size, 128, H/8, W/8)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),                          # (batch_size, 128, H/8, W/8)
            conv_dw_no_bn(out_channels, out_channels),                          # (batch_size, 128, H/8, W/8)
            conv_dw_no_bn(out_channels, out_channels)                           # (batch_size, 128, H/8, W/8)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)                                                       # 입력: (batch_size, 512, H/8, W/8)   # align 후: (batch_size, 128, H/8, W/8)
        x = self.conv(x + self.trunk(x))                                        # (batch_size, 128, H/8, W/8)
        return x

class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2 ,padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features

class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stage=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32, stride=2, bias=False),          #(batch_size, 3, H, W),       #(batch_size, 32, H/2, W/2)
            conv_dw(32, 64),                            #(batch_size, 32, H/2, W/2)   #(batch_size, 64, H/2, W/2)
            conv_dw(64, 128, stride=2),                 #(batch_size, 64, H/2, W/2_   #(batch_size, 128, H/4, W/4)
            conv_dw(128, 128),                          #(batch_size, 128, H/4, W/4)  #(batch_size, 128, H/4, W/4)
            conv_dw(128, 256, stride=2),                #(batch_size, 128, H/4, W/4)  #(batch_size, 256, H/8, W/8)
            conv_dw(256, 256),                          #(batch_size, 256, H/8, W/8)  #(batch_size, 256, H/8, W/8)
            conv_dw(256, 512),  # conv4_2               #(batch_size, 256, H/8, W/8)  #(batch_size, 512, H/8, W/8)
            conv_dw(512, 512, dilation=2, padding=2),   #(batch_size, 512, H/8, W/8)  #(batch_size, 512, H/8, W/8)
            conv_dw(512, 512),                          #(batch_size, 512, H/8, W/8)
            conv_dw(512, 512),                                     #:
            conv_dw(512, 512),                                     #:
            conv_dw(512, 512)  # conv5_5                #(batch_size, 512, H/8, W/8)
        )
        self.cpm = Cpm(512, num_channels)
        # 입력: (batch_size, 512, H/8, W/8)
        # align 후: (batch_size, 128, H/8, W/8)
        # trunk: (batch_size, 128, H/8, W/8)
        # conv 출력: (batch_size, 128, H/8, W/8)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        # 입력: (batch_size, 128, H/8, W/8), (num_channels=128, num_heatmaps=19, num_pafs=38)
        # trunk 출력: (batch_size, 128, H/8, W/8)
        # heatmaps: (batch_size, 19, H/8, W/8)
        # pafs: (batch_size, 38, H/8, W/8)

        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stage):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels, num_heatmaps, num_pafs))


    def forward(self, x):
        backbone_features = self.model(x)
        # backbone_features: (batch_size, 512, H/8, W/8)

        backbone_features = self.cpm(backbone_features)
        # backbone_features: (batch_size, 128, H/8, W/8)

        stages_output = self.initial_stage(backbone_features)
        # heatmaps: (batch_size, 19, H/8, W/8)
        # pafs: (batch_size, 38, H/8, W/8)

        for refinement_stage in self.refinement_stages:
            stages_output.extend(refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
            # backbone_feature (batch_size, 128, H/8, W/8)
            # stages_output[-2] == # heatmaps: (batch_size, 19, H/8, W/8)
            # stages_output[-1] == # pafs: (batch_size, 38, H/8, W/8)
            # final(concat) == # (batch_size, 185(128+19+38), H/8, W/8)
        # each refinement step:
        # input: (batch_size, 185, H/8, W/8)
        # output heatmaps: (batch_size, 19, H/8, W/8)
        # output pafs: (batch_size, 38, H/8, W/8)

        return stages_output