import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class InceptionV2(nn.Module):
    def __init__(self, aux_logits=True):
        super(InceptionV2, self).__init__()
        
        self.aux_logits = aux_logits
        
        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        
        conv1out, conv2out, conv3out, conv4out, conv5out = 64, 64, 64, 96, 192
        ch71, out71, out11 = 128, 128, 128
        ch72, out72, out12 = 128, 128, 128
        ch73, out73, out13 = 128, 192, 192
        ch31, ch32, out31, ch33, out32, out14, out15 = 64, 64, 64, 64, 48, 96, 96
        ch74, out74, out16 = 128, 128, 128
        ch75, out75, out17 = 128, 128, 128
        ch76, out76, out18 = 128, 192, 192
        ch77, out77, out19 = 128, 192, 192
        ch34, ch35, out33, ch36, out34, out20, out21 = 64, 64, 64, 64, 48, 96, 96
        ch37, ch38, out35, out36, out22, out23 = 384, 384, 384, 384, 256, 256
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv1out, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=conv1out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1out, out_channels=conv2out, kernel_size=3),
            nn.BatchNorm2d(num_features=conv2out),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=conv2out, out_channels=conv3out, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=conv3out),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=conv3out, out_channels=conv4out, kernel_size=1),
            nn.BatchNorm2d(num_features=conv4out),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=conv4out, out_channels=conv5out, kernel_size=3),
            nn.BatchNorm2d(num_features=conv5out),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception1 = InceptionB(in_channels=conv5out, ch7=ch71, out7=out71, out1=out11)
        self.inception2 = InceptionB(in_channels=(2*out71 + 2*out11), ch7=ch72, out7=out72, out1=out12)
        self.inception3 = InceptionB(in_channels=(2*out72 + 2*out12), ch7=ch73, out7=out73, out1=out13)
        self.inception4 = InceptionA(in_channels=(2*out73 + 2*out13), to3x3_1=ch31, to3x3_2=ch32, out3x3_1=out31, 
                                     to3x3_3=ch33, out3x3_2=out32, out1x1_1=out14, out1x1_2=out15)
        self.inception5 = InceptionB(in_channels=(out31+out32+out14+out15), ch7=ch74, out7=out74, out1=out16)
        self.inception6 = InceptionD(in_channels=(2*out74 + 2*out16), ch7=ch75, out7=out75, out1=out17)
        self.inception7 = InceptionD(in_channels=(2*out75 + 2*out17), ch7=ch76, out7=out76, out1=out18)
        self.inception8 = InceptionD(in_channels=(2*out76 + 2*out18), ch7=ch77, out7=out77, out1=out19)
        
        if self.aux_logits:
            self.aux = InceptionAux(in_channels=(2*out77 + 2*out19), num_classes=num_classes)
        else:
            self.aux = None
            
        self.inception9 = InceptionA(in_channels=(2*out77 + 2*out19), to3x3_1=ch34, to3x3_2=ch35, out3x3_1=out33, 
                                     to3x3_3=ch36, out3x3_2=out34, out1x1_1=out20, out1x1_2=out21)
        self.inception10 = InceptionC(in_channels=(out33+out34+out20+out21), ch3=ch37, out3=out35, out1=out22)
        self.inception11 = InceptionC(in_channels=(4*out35 + 2*out22), ch3=ch38, out3=out36, out1=out23)
        
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=(4*out36 + 2*out23), out_features=num_classes)
        self.dropout = nn.Dropout(p=cfg.drop_cnn)
        
    def forward(self, x):                
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.inception8(x)
        if self.aux is not None:
            if self.training:
                aux = self.aux(x)
                
        x = self.inception9(x)
        x = self.inception10(x)
        x = self.inception11(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return (x, aux)
        else:
            return x
                

class InceptionA(nn.Module):
    def __init__(self, in_channels, to3x3_1, to3x3_2, out3x3_1, to3x3_3, out3x3_2, out1x1_1, out1x1_2):
        super(InceptionA, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=to3x3_1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=to3x3_1),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3_1, out_channels=to3x3_2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=to3x3_2),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3_2, out_channels=out3x3_1, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out3x3_1),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=to3x3_3, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=to3x3_3),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3_3, out_channels=out3x3_2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out3x3_2),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1_1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1x1_1),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1_2, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1x1_2),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
        

class InceptionB(nn.Module):
    def __init__(self, in_channels, ch7, out7, out1):
        super(InceptionB, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch7, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=out7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(num_features=out7),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch7, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=out7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(num_features=out7),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, ch3, out3, out1):
        super(InceptionC, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch3, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch3),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch3, out_channels=ch3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=ch3),
            nn.ReLU()
        )
        
        self.branch1a = nn.Sequential(
            nn.Conv2d(in_channels=ch3, out_channels=out3, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=out3),
            nn.ReLU()
        )
        
        self.branch1b = nn.Sequential(
            nn.Conv2d(in_channels=ch3, out_channels=out3, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=out3),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch3, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch3),
            nn.ReLU()
        )
        
        self.branch2a = nn.Sequential(
            nn.Conv2d(in_channels=ch3, out_channels=out3, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=out3),
            nn.ReLU()
        )
        
        self.branch2b = nn.Sequential(
            nn.Conv2d(in_channels=ch3, out_channels=out3, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=out3),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch1 = torch.cat([self.branch1a(branch1), self.branch1b(branch1)], dim=1)
        branch2 = self.branch2(x)
        branch2 = torch.cat([self.branch2a(branch2), self.branch2b(branch2)], dim=1)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
    

class InceptionD(nn.Module):
    def __init__(self, in_channels, ch7, out7, out1):
        super(InceptionD, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch7, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),            
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=out7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=out7),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch7, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=ch7, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch7),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch7, out_channels=out7, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=out7),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(num_features=out1),
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
        
        auxout1 = 512
        auxout2 = 768
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=auxout1, kernel_size=1),
            nn.BatchNorm2d(auxout1),
            nn.ReLU(),
            nn.Conv2d(in_channels=auxout1, out_channels=auxout2, kernel_size=5),
            nn.BatchNorm2d(auxout2),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(auxout2, cfg.head_size1)
        self.fc2 = nn.Linear(cfg.head_size1, num_classes)
        self.dropout = nn.Dropout(p=cfg.drop_fc)
        
    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
