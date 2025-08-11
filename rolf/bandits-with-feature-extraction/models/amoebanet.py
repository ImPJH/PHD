import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()


def reduction_out(in_channels, out_channels):
    return 4*in_channels + out_channels

def normal_out(in_channels, out1, out2):
    return in_channels + out1 + out2


class AmoebaNet(nn.Module):
    def __init__(self) -> None:
        super(AmoebaNet, self).__init__()

        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        
        outfeat1 = 24
        middle1, middle2, middle3, middle4, middle5 = [4 for _ in range(1, 5+1)]
        middle6, middle7, middle8, middle9, middle0 = [4 for _ in range(6, 10+1)]
        out1, out2, out3, out4 = [4 for _ in range(1, 4+1)]
        out5, out6, out7, out8 = [4 for _ in range(1, 4+1)]
        out9, out0, outa, outb = [4 for _ in range(1, 4+1)]
        outc, outd, oute, outf = [4 for _ in range(1, 4+1)]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=outfeat1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outfeat1),
            nn.ReLU()
        )
        
        self.reduction1 = ReductionCell(in_channels=outfeat1, middle=middle1, sepout6=out1)
        reduction1_out = reduction_out(outfeat1, out1)
        self.reduction2 = ReductionCell(in_channels=reduction1_out, middle=middle2, sepout6=out2)
        reduction2_out = reduction_out(reduction1_out, out2)
        
        self.normal1 = NormalCell(in_channels=reduction2_out, middle=middle3, sepout4=out3, sepout5=out4)
        normal1_out = normal_out(reduction2_out, out3, out4)
        self.normal2 = NormalCell(in_channels=normal1_out, middle=middle4, sepout4=out5, sepout5=out6)
        normal2_out = normal_out(normal1_out, out5, out6)
        
        self.reduction3 = ReductionCell(in_channels=normal2_out, middle=middle5, sepout6=out7)
        reduction3_out = reduction_out(normal2_out, out7)
        
        self.normal3 = NormalCell(in_channels=reduction3_out, middle=middle6, sepout4=out8, sepout5=out9)
        normal3_out = normal_out(reduction3_out, out8, out9)
        self.normal4 = NormalCell(in_channels=normal3_out, middle=middle7, sepout4=out0, sepout5=outa)
        normal4_out = normal_out(normal3_out, out0, outa)
        
        self.reduction4 = ReductionCell(in_channels=normal4_out, middle=middle8, sepout6=outb)
        reduction4_out = reduction_out(normal4_out, outb)
        
        self.normal5 = NormalCell(in_channels=reduction4_out, middle=middle9, sepout4=outc, sepout5=outd)
        normal5_out = normal_out(reduction4_out, outc, outd)
        self.normal6 = NormalCell(in_channels=normal5_out, middle=middle0, sepout4=oute, sepout5=outf)
        normal6_out = normal_out(normal5_out, oute, outf)
        
        self.adaptivepool = nn.AdaptiveAvgPool2d([1, 1])
        
        if cfg.multihead:
            self.fc = nn.Sequential(
                nn.Linear(normal6_out, cfg.head_size1),
                nn.BatchNorm1d(num_features=cfg.head_size1),
                nn.ReLU(),
                nn.Dropout(p=cfg.drop_fc),
                nn.Linear(cfg.head_size1, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(p=cfg.drop_fc),
                nn.Linear(normal6_out, num_classes),
            )
            
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.reduction1(x)
        x = self.reduction2(x)
        
        x = self.normal1(x)
        x = self.normal2(x)
        
        x = self.reduction3(x)
        
        x = self.normal3(x)
        x = self.normal4(x)
        
        x = self.reduction4(x)
        
        x = self.normal5(x)
        x = self.normal6(x)
        
        x = self.adaptivepool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class NormalCell(nn.Module):
    def __init__(self, in_channels, middle, sepout4, sepout5):
        super(NormalCell, self).__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)    # for 2nd block
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)    # for 3rd block 
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)    # for 6th block
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=sepout5, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=sepout5)
        )
        
        self.sep3_4 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout4, kernel_size=3, padding_size1=1, stride1=1, padding_size2=1, stride2=1)
        self.sep5_4 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout4, kernel_size=5, padding_size1=2, stride1=1, padding_size2=2, stride2=1)
        self.sep3_5 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout5, kernel_size=3, padding_size1=1, stride1=1, padding_size2=1, stride2=1)
        self.sep3_6 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout4, kernel_size=3, padding_size1=1, stride1=1, padding_size2=1, stride2=1)
    
    def forward(self, x):
        # x_ = x.clone().detach()
        
        ## 2nd block
        avg1 = self.avgpool1(x)
        max1 = self.maxpool(x)
        branch1 = avg1 + max1
        branch1 = F.relu(branch1)
        
        ## 3rd block
        identity1 = self.downsample1(x)
        avg2 = self.avgpool2(x)
        branch2 = identity1 + avg2
        branch2 = F.relu(branch2)
        
        ## 4th block
        sep1 = self.sep5_4(branch1)
        sep2 = self.sep3_4(x)
        branch3 = sep1 + sep2
        branch3 = F.relu(branch3)
        
        ## 5th block
        sep3 = self.sep3_5(branch1)
        identity2 = self.downsample2(x)
        branch4 = sep3 + identity2
        branch4 = F.relu(branch4)
        
        ## 6th block
        avg3 = self.avgpool3(branch3)
        sep4 = self.sep3_6(x)
        branch5 = avg3 + sep4
        branch5 = F.relu(branch5)
        
        ## output
        output = torch.cat([branch2, branch4, branch5], dim=1) # in_channels + sepout5 + sepout4
        return output
        

class ReductionCell(nn.Module):
    def __init__(self, in_channels, middle, sepout6):
        super(ReductionCell, self).__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)    # for 2nd block
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)    # for 3rd block 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)    # for 6th block
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
                
        self.sep3_2 = SepBlock(in_channels=in_channels, middle=middle, out_channels=in_channels, kernel_size=3, padding_size1=1, stride1=1, padding_size2=1, stride2=2)
        self.sep3_6 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout6, kernel_size=3, padding_size1=1, stride1=1, padding_size2=1, stride2=2)
        self.sep7_4 = SepBlock(in_channels=in_channels, middle=middle, out_channels=in_channels, kernel_size=7, padding_size1=3, stride1=1, padding_size2=3, stride2=2)
        self.sep7_5 = SepBlock(in_channels=in_channels, middle=middle, out_channels=in_channels, kernel_size=7, padding_size1=3, stride1=1, padding_size2=3, stride2=2)
        self.sep7_6 = SepBlock(in_channels=in_channels, middle=middle, out_channels=sepout6, kernel_size=7, padding_size1=3, stride1=1, padding_size2=3, stride2=2)
        
    def forward(self, x):
        ## 2nd block
        avg1 = self.avgpool1(x)
        sep1 = self.sep3_2(x)
        branch1 = F.relu(avg1 + sep1)
        
        ## 3rd block
        max1 = self.maxpool1(x)
        max2 = self.maxpool2(x)
        branch2 = F.relu(max1 + max2)
        
        ## 4th block
        max3 = self.maxpool3(x)
        sep2 = self.sep7_4(x)
        branch3 = F.relu(max3 + sep2)
        
        ## 5th block
        sep3 = self.sep7_5(x)
        avg2 = self.avgpool2(x)
        branch4 = F.relu(sep3 + avg2)
        
        ## 6th block
        sep4 = self.sep3_6(x)
        sep5 = self.sep7_6(x)
        branch5 = F.relu(sep4 + sep5)
        
        return torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)  # in_channels + in_channels + in_channels + in_channels + sepout6
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_feautres=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x
        

class SepBlock(nn.Module):
    def __init__(self, in_channels, middle, out_channels, kernel_size, padding_size2, stride2, padding_size1=1, stride1=1):
        super(SepBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=middle, kernel_size=(1, kernel_size), padding=(0, padding_size1), stride=stride1),
            nn.BatchNorm2d(num_features=middle),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle, out_channels=out_channels, kernel_size=(kernel_size, 1), padding=(padding_size2, 0), stride=stride2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.layer(x)
        return x
