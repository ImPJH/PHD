import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class AlexNet(nn.Module):
    def __init__(self):        
        super(AlexNet, self).__init__()
        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100

        ## from 4 ~ 2048
        feats = [4, 8, 16, 32, 64, 
                 96, 128, 192, 256, 384, 
                 512, 768, 1024, 1536, 2048]
        
        feat1 = feats[5]
        feat2 = feats[8] 
        feat3 = feats[9] 
        feat4 = feats[9]
        feat5 = feats[8]
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = feat1, kernel_size = 11, stride = 4, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = feat1, out_channels = feat2, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = feat2, out_channels = feat3, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = feat3, out_channels = feat4, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = feat4, out_channels = feat5, kernel_size = 3, padding = 1)
        
        self.num_feat = 5 * 5 * feat5
        
        self.fc1 = nn.Linear(self.num_feat, cfg.head_size1)
        self.fc2 = nn.Linear(cfg.head_size1, cfg.head_size2)
        self.fc3 = nn.Linear(cfg.head_size2, num_classes)

        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv_bn1 = nn.BatchNorm2d(feat1)
        self.conv_bn2 = nn.BatchNorm2d(feat2)
        self.conv_bn3 = nn.BatchNorm2d(feat3)
        self.conv_bn4 = nn.BatchNorm2d(feat4)
        self.conv_bn5 = nn.BatchNorm2d(feat5)
        
        self.fc_bn1 = nn.BatchNorm1d(cfg.head_size1)
        self.fc_bn2 = nn.BatchNorm1d(cfg.head_size2)
        # self.fc_bn3 = nn.BatchNorm1d(cfg.head_size2)
        
        self.dropout = nn.Dropout(p=cfg.drop_fc)
    
    def forward(self, x):
        ## CNN
        x = self.conv1(x)
        x = F.relu(self.conv_bn1(x))
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(self.conv_bn2(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(self.conv_bn3(x))       
        x = self.conv4(x)
        x = F.relu(self.conv_bn4(x))
        x = self.conv5(x)
        x = F.relu(self.conv_bn5(x))
        x = self.pool(x)
        
        ## head
        x = x.view(-1, self.num_feat)
        x = self.fc1(x)
        x = F.relu(self.fc_bn1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc_bn2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
