import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inc, outc, midc=None):
        super(Conv, self).__init__()
        if midc is None:
            midc = outc
        self.conv = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, padding=1),
            nn.BatchNorm2d(midc),   #归一化
            nn.ReLU(inplace=True),  #inplace=true：直接在原来的内存地址上修改数据，节省显存
            nn.Conv2d(midc, outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return  self.conv(x)

class downs(nn.Module):
    def __init__(self, inc, outc):
        super(downs, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(inc, outc)
        )
    
    def forward(self, x):
        return self.down(x)

class ups(nn.Module):
    def __init__(self, inc, outc):
        super(ups, self).__init__()
        self.up = nn.ConvTranspose2d(inc, inc//2, kernel_size=2, stride=2)
        self.conv = Conv(inc, outc)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 补像素，对其x1, x2的尺寸
        diffY = x2.size()[2] - x1.size()[2] # H
        diffX = x2.size()[3] - x1.size()[3] # W
        # 参数 list = [pad_left, pad_right, pad_top, pad_bottom]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x =  torch.cat([x2, x1], dim=1) #拼接
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, inc, outc):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, inc=3, outc=1):
        super(UNet, self).__init__()

        self.inp = Conv(inc, 64)
        self.down1 = downs(64, 128)
        self.down2 = downs(128, 256)
        self.down3 = downs(256, 512)
        self.down4 = downs(512, 1024)

        self.up1 = ups(1024, 512)
        self.up2 = ups(512, 256)
        self.up3 = ups(256, 128)
        self.up4 = ups(128, 64)

        self.out = outconv(64, outc)

    def forward(self, x):
        x1 = self.inp(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)

        return torch.sigmoid(logits)

'''
if __name__ == "__main__":
    model = UNet(3, 1)
    x = torch.randn(4, 3, 512, 512)
    y = model(x)
    print(x.shape)
    print(y.shape)
'''
        
