import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class GoogLeResNet(nn.Module):
    def __init__(self, aux_logits=True):
        super(GoogLeResNet, self).__init__()

        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
            
        self.aux_logits = aux_logits
        
        outfeat1, outfeat2, itm, outfeat3 = 128, 128, 128, 128
        out11, out31, out51, outpool1, to31, to51 = 64, 64, 64, 64, 128, 128
        out12, out32, out52, outpool2, to32, to52 = 64, 64, 64, 64, 128, 128
        out13, out33, out53, outpool3, to33, to53 = 96, 96, 96, 96, 128, 128
        out14, out34, out54, outpool4, to34, to54 = 96, 96, 96, 96, 128, 128
        out15, out35, out55, outpool5, to35, to55 = 128, 128, 128, 128, 128, 128
        out16, out36, out56, outpool6, to36, to56 = 128, 128, 128, 128, 128, 128
        out17, out37, out57, outpool7, to37, to57 = 128, 128, 128, 128, 128, 128
        out18, out38, out58, outpool8, to38, to58 = 128, 128, 128, 128, 128, 128
        out19, out39, out59, outpool9, to39, to59 = 256, 256, 256, 256, 256, 256
        out10, out30, out50, outpool0, to30, to50 = 256, 256, 256, 256, 256, 256
        out1a, out3a, out5a, outpoola, to3a, to5a = 256, 256, 256, 256, 256, 256
        out1b, out3b, out5b, outpoolb, to3b, to5b = 384, 384, 384, 384, 256, 256
        out1c, out3c, out5c, outpoolc, to3c, to5c = 384, 384, 384, 384, 256, 256
        out1d, out3d, out5d, outpoold, to3d, to5d = 512, 512, 512, 512, 256, 256
        out1e, out3e, out5e, outpoole, to3e, to5e = 512, 512, 512, 512, 256, 256
        out1f, out3f, out5f, outpoolf, to3f, to5f = 512, 512, 512, 512, 256, 256
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=outfeat1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(outfeat1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=outfeat1, out_channels=outfeat2, kernel_size=1),
            nn.BatchNorm2d(outfeat2),
            nn.ReLU()
        )
        self.res = ResBlock(in_channels=outfeat2, intermediate=itm, out_channels=outfeat3)
        
        self.inception1 = Inception(in_channels=outfeat3, out1x1=out11, to3x3=to31, out3x3=out31, to5x5=to51, out5x5=out51, poolout=outpool1, modified=cfg.google_modified)
        self.inception2 = Inception(in_channels=out11+out31+out51+outpool1, out1x1=out12, to3x3=to32, out3x3=out32, to5x5=to52, out5x5=out52, poolout=outpool2, modified=cfg.google_modified)
        self.inception3 = Inception(in_channels=out12+out32+out52+outpool2, out1x1=out13, to3x3=to33, out3x3=out33, to5x5=to53, out5x5=out53, poolout=outpool3, modified=cfg.google_modified)
        self.inception4 = Inception(in_channels=out13+out33+out53+outpool3, out1x1=out14, to3x3=to34, out3x3=out34, to5x5=to54, out5x5=out54, poolout=outpool4, modified=cfg.google_modified)
        self.inception5 = Inception(in_channels=out14+out34+out54+outpool4, out1x1=out15, to3x3=to35, out3x3=out35, to5x5=to55, out5x5=out55, poolout=outpool5, modified=cfg.google_modified)
        self.inception6 = Inception(in_channels=out15+out35+out55+outpool5, out1x1=out16, to3x3=to36, out3x3=out36, to5x5=to56, out5x5=out56, poolout=outpool6, modified=cfg.google_modified)
        self.inception7 = Inception(in_channels=out16+out36+out56+outpool6, out1x1=out17, to3x3=to37, out3x3=out37, to5x5=to57, out5x5=out57, poolout=outpool7, modified=cfg.google_modified)
        self.inception8 = Inception(in_channels=out17+out37+out57+outpool7, out1x1=out18, to3x3=to38, out3x3=out38, to5x5=to58, out5x5=out58, poolout=outpool8, modified=cfg.google_modified)
        self.inception9 = Inception(in_channels=out18+out38+out58+outpool8, out1x1=out19, to3x3=to39, out3x3=out39, to5x5=to59, out5x5=out59, poolout=outpool9, modified=cfg.google_modified)
        self.inception0 = Inception(in_channels=out19+out39+out59+outpool9, out1x1=out10, to3x3=to30, out3x3=out30, to5x5=to50, out5x5=out50, poolout=outpool0, modified=cfg.google_modified)
        self.inceptiona = Inception(in_channels=out10+out30+out50+outpool0, out1x1=out1a, to3x3=to3a, out3x3=out3a, to5x5=to5a, out5x5=out5a, poolout=outpoola, modified=cfg.google_modified)
        self.inceptionb = Inception(in_channels=out1a+out3a+out5a+outpoola, out1x1=out1b, to3x3=to3b, out3x3=out3b, to5x5=to5b, out5x5=out5b, poolout=outpoolb, modified=cfg.google_modified)
        self.inceptionc = Inception(in_channels=out1b+out3b+out5b+outpoolb, out1x1=out1c, to3x3=to3c, out3x3=out3c, to5x5=to5c, out5x5=out5c, poolout=outpoolc, modified=cfg.google_modified)
        self.inceptiond = Inception(in_channels=out1c+out3c+out5c+outpoolc, out1x1=out1d, to3x3=to3d, out3x3=out3d, to5x5=to5d, out5x5=out5d, poolout=outpoold, modified=cfg.google_modified)
        self.inceptione = Inception(in_channels=out1d+out3d+out5d+outpoold, out1x1=out1e, to3x3=to3e, out3x3=out3e, to5x5=to5e, out5x5=out5e, poolout=outpoole, modified=cfg.google_modified)
        self.inceptionf = Inception(in_channels=out1e+out3e+out5e+outpoole, out1x1=out1f, to3x3=to3f, out3x3=out3f, to5x5=to5f, out5x5=out5f, poolout=outpoolf, modified=cfg.google_modified)
        
        if aux_logits:
            self.aux1 = InceptionAux(in_channels=out16+out36+out56+outpool6, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=out10+out30+out50+outpool0, num_classes=num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        
        if cfg.multihead:
            self.fc = nn.Sequential(
                nn.Linear(out1f+out3f+out5f+outpoolf, cfg.head_size1),
                nn.BatchNorm1d(num_features=cfg.head_size1),
                nn.ReLU(),
                nn.Dropout(p=cfg.drop_fc),
                nn.Linear(cfg.head_size1, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(p=cfg.drop_fc),
                nn.Linear(out1f+out3f+out5f+outpoolf, num_classes),
            )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.res(x)
        x = self.maxpool2(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.maxpool3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
                
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.inception0(x)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
                
        x = self.inceptiona(x)
        x = self.inceptionb(x)
        x = self.inceptionc(x)
        x = self.maxpool4(x)
        x = self.inceptiond(x)
        x = self.inceptione(x)
        x = self.inceptionf(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return (x, aux2, aux1)
        else:
            return x
        

class Inception(nn.Module):
    def __init__(self, in_channels, out1x1, to3x3, out3x3, to5x5, out5x5, poolout, modified=False):
        super(Inception, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1, eps=0.001),
            # nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=to3x3, kernel_size=1),
            nn.BatchNorm2d(to3x3, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3, out_channels=out3x3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out3x3, eps=0.001),
            # nn.ReLU()
        )
        
        if modified:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=3, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                # nn.ReLU()
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=5, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                # nn.ReLU()
            )
            
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=in_channels, out_channels=poolout, kernel_size=1),
            nn.BatchNorm2d(poolout, eps=0.001),
            # nn.ReLU()
        )
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out1x1, eps=0.001)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out1x1, eps=0.001)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out1x1, eps=0.001)
        )

        self.downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=poolout, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out1x1, eps=0.001)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)   
        downsample1 = self.downsample1(x)
        identity1 = branch1 + downsample1
        # identity1 = F.relu(branch1+downsample1)

        branch2 = self.branch2(x)
        downsample2 = self.downsample2(x)
        identity2 = branch2 + downsample2
        # identity2 = F.relu(branch2+downsample2)
        
        branch3 = self.branch3(x)
        downsample3 = self.downsample3(x)
        identity3 = branch3 + downsample3
        # identity3 = F.relu(branch3+downsample3)

        branch4 = self.branch4(x)
        downsample4 = self.downsample4(x)
        identity4 = branch4 + downsample4
        # identity4 = F.relu(branch4+downsample4)
        
        branch_out = torch.cat([identity1, identity2, identity3, identity4], dim=1)
        output = F.relu(branch_out)
        
        return output
    

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        auxout = 512
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=auxout, kernel_size=1),
            nn.BatchNorm2d(auxout),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(4*4*auxout, cfg.head_size1)
        self.fc2 = nn.Linear(cfg.head_size1, num_classes)
        self.dropout = nn.Dropout(p=cfg.drop_fc)
        
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
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
