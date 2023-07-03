import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class ResNet18(nn.Module):
    def __init__(self):        
        super(ResNet18, self).__init__()
        
        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        
        feat1, feat2, feat3, feat4, feat5 = 64, 64, 64, 64, 64
        feat6, feat7, feat8, feat9 = 128, 128, 128, 128
        feat10, feat11, feat12, feat13 = 256, 256, 256, 256
        feat14, feat15, feat16, feat17 = 512, 512, 512, 512

        if cfg.decompose:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=feat1, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=feat1),
                nn.ReLU(),
                nn.Conv2d(in_channels=feat1, out_channels=feat1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=feat1),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=feat1, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_features=feat1),
                nn.ReLU()
            )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res1 = ResBlock(in_channels=feat1, intermediate=feat2, out_channels=feat3)
        self.res2 = ResBlock(in_channels=feat3, intermediate=feat4, out_channels=feat5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res3 = ResBlock(in_channels=feat5, intermediate=feat6, out_channels=feat7, shortcut=cfg.shortcut)
        self.res4 = ResBlock(in_channels=feat7, intermediate=feat8, out_channels=feat9)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res5 = ResBlock(in_channels=feat9, intermediate=feat10, out_channels=feat11, shortcut=cfg.shortcut)
        self.res6 = ResBlock(in_channels=feat11, intermediate=feat12, out_channels=feat13)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res7 = ResBlock(in_channels=feat13, intermediate=feat14, out_channels=feat15, shortcut=cfg.shortcut)
        self.res8 = ResBlock(in_channels=feat15, intermediate=feat16, out_channels=feat17)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(feat17, out_features=cfg.head_size1),
            nn.Dropout(p=cfg.drop_fc),
            nn.Linear(cfg.head_size1, out_features=num_classes)
        )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool2(x)
        
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool3(x)
        
        x = self.res5(x)
        x = self.res6(x)
        x = self.pool4(x)
        
        x = self.res7(x)
        x = self.res8(x)
        
        ## head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
                
        return x
   

class ResBlock(nn.Module):
    def __init__(self, in_channels, intermediate, out_channels, shortcut=True):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=intermediate),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
        self.shortcut = shortcut

    def forward(self, x):
        branch = self.block(x)
        if self.shortcut:
            identity = self.downsample(x)
            x = branch + identity
        else:
            x = branch
        x = F.relu(x)
        return x
