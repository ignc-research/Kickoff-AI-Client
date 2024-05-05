import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from pointnet2.models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes=3, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fully_connect1 = nn.Linear(1024, 512)
        self.fully_connect2 = nn.Linear(512, 1)

    def forward(self, in1, in2,training=False):
        # Set Abstraction layers
        if training:
            l0_points = in1
            l0_xyz = in1
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

            l0_points2 = in2
            l0_xyz2 = in2
            l1_xyz2, l1_points2 = self.sa1(l0_xyz2, l0_points2)
            l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)
            l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)

            x = torch.abs(l3_points - l3_points2)
            # x = torch.cat([l3_points,l3_points2],dim=1)
            x = x.view(1, -1)
            x = self.fully_connect1(x)
            x = self.fully_connect2(x)
            return x
        else:
    # def forward(self, in1,in2):
    #     # Set Abstraction layers
    #
            l0_points = in1
            l0_xyz = in1
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)


            x_list = []
            for i in range(1,len(in2)):
                tic=time.time()
                point2=torch.unsqueeze(in2[i],dim=0)
                l0_points2 = point2
                l0_xyz2 = point2
                l1_xyz2, l1_points2 = self.sa1(l0_xyz2, l0_points2)
                l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)
                l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)



                x = torch.abs(l3_points - l3_points2)
                # x = torch.cat([l3_points,point2],dim=1)
                x = x.view(1,-1)
                x = self.fully_connect1(x)
                x = self.fully_connect2(x)
                x_list.append(x)
            return x_list

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss