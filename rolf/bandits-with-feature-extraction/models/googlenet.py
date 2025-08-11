import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

# feats = [4, 8, 16, 32, 64, 
#          96, 128, 192, 256, 384, 
#          512, 768, 1024, 1536, 2048]

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True):
        super(GoogLeNet, self).__init__()

        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
            
        self.aux_logits = aux_logits
        
        outfeat1, outfeat2, outfeat3 = 256, 384, 512
        out11, out12, out13, out14, out15, out16, out17, out18, out19 = 256, 256, 384, 384, 384, 512, 512, 512, 512
        to31, to32, to33, to34, to35, to36, to37, to38, to39 = 256, 256, 256, 384, 384, 384, 512, 512, 512
        out31, out32, out33, out34, out35, out36, out37, out38, out39 = 384, 384, 384, 512, 512, 512, 1024, 1024, 1024
        to51, to52, to53, to54, to55, to56, to57, to58, to59 = 128, 128, 128, 192, 192, 192, 256, 256, 256
        out51, out52, out53, out54, out55, out56, out57, out58, out59 = 384, 384, 384, 512, 512, 512, 768, 768, 768
        outpool1, outpool2, outpool3, outpool4, outpool5, outpool6, outpool7, outpool8, outpool9 = 96, 96, 128, 128, 128, 128, 192, 192, 192 
        
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=outfeat2, out_channels=outfeat3, kernel_size=3, padding='same')
        )
        
        self.inception1 = Inception(in_channels=outfeat3, out1x1=out11, to3x3=to31, out3x3=out31, to5x5=to51, out5x5=out51, poolout=outpool1, modified=cfg.google_modified)
        self.inception2 = Inception(in_channels=out11+out31+out51+outpool1, out1x1=out12, to3x3=to32, out3x3=out32, to5x5=to52, out5x5=out52, poolout=outpool2, modified=cfg.google_modified)
        self.inception3 = Inception(in_channels=out12+out32+out52+outpool2, out1x1=out13, to3x3=to33, out3x3=out33, to5x5=to53, out5x5=out53, poolout=outpool3, modified=cfg.google_modified)
        self.inception4 = Inception(in_channels=out13+out33+out53+outpool3, out1x1=out14, to3x3=to34, out3x3=out34, to5x5=to54, out5x5=out54, poolout=outpool4, modified=cfg.google_modified)
        self.inception5 = Inception(in_channels=out14+out34+out54+outpool4, out1x1=out15, to3x3=to35, out3x3=out35, to5x5=to55, out5x5=out55, poolout=outpool5, modified=cfg.google_modified)
        self.inception6 = Inception(in_channels=out15+out35+out55+outpool5, out1x1=out16, to3x3=to36, out3x3=out36, to5x5=to56, out5x5=out56, poolout=outpool6, modified=cfg.google_modified)
        self.inception7 = Inception(in_channels=out16+out36+out56+outpool6, out1x1=out17, to3x3=to37, out3x3=out37, to5x5=to57, out5x5=out57, poolout=outpool7, modified=cfg.google_modified)
        self.inception8 = Inception(in_channels=out17+out37+out57+outpool7, out1x1=out18, to3x3=to38, out3x3=out38, to5x5=to58, out5x5=out58, poolout=outpool8, modified=cfg.google_modified)
        self.inception9 = Inception(in_channels=out18+out38+out58+outpool8, out1x1=out19, to3x3=to39, out3x3=out39, to5x5=to59, out5x5=out59, poolout=outpool9, modified=cfg.google_modified)
        
        if aux_logits:
            self.aux1 = InceptionAux(in_channels=out13+out33+out53+outpool3, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=out16+out36+out56+outpool6, num_classes=num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.dropout = nn.Dropout(p=cfg.drop_fc)
        self.fc = nn.Linear(out19+out39+out59+outpool9, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool3(x)
        x = self.inception3(x)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
                
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
                
        x = self.inception7(x)
        x = self.maxpool4(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=to3x3, kernel_size=1),
            nn.BatchNorm2d(to3x3, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3, out_channels=out3x3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out3x3, eps=0.001),
            nn.ReLU()
        )
        
        if modified:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=3, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                nn.ReLU()
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=5, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                nn.ReLU()
            )
            
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=in_channels, out_channels=poolout, kernel_size=1),
            nn.BatchNorm2d(poolout, eps=0.001),
            nn.ReLU()            
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
    

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
        