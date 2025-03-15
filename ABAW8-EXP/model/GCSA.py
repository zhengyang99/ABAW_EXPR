import torch.nn as nn
import torch


class GCSA(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GCSA, self).__init__()

       
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)), 
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels) 
        )

        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True), 
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(in_channels)  
        )

  
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
    
        x = x.view(batchsize, groups, channels_per_group, height, width)
      
        x = torch.transpose(x, 1, 2).contiguous()
     
        x = x.view(batchsize, -1, height, width)

        return x

 
    def forward(self, x):
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c) 
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  

        x = x * x_channel_att  

        x = self.channel_shuffle(x, groups=4) 

        x_spatial_att = self.spatial_attention(x).sigmoid()  

        out = x * x_spatial_att  

        return out  


if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)  
    b, c, h, w = x.shape  
    net = GCSA(in_channels=c) 
    y = net(x)  
    print(y.size())  
