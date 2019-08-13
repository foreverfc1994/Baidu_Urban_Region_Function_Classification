import torch
import torch.nn as nn
import torch.nn.functional as F
class Residual(nn.Module):
    def __init__(self, channel, padding = 1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channel)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channel)
        self.downsample = nn.Conv2d(channel, 2 * channel , kernel_size=3, stride=2, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(num_features=2 * channel)
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.downsample(self.relu(input + x))
        return self.relu(self.bn3(x))

class BasicConv(nn.Module):
    def __init__(self, in_channels, relu = True):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(num_features=in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), stride=1),
            nn.BatchNorm3d(num_features=in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=1),
            nn.BatchNorm3d(num_features=in_channels)
        )
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
    def forward(self, input):
        x = self.conv(input)
        x = x + input
        if self.relu:
            x = self.relu(x)
        return x

class Residual3(nn.Module):
    def __init__(self, in_channels):
        super(Residual3, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channels=in_channels),
            BasicConv(in_channels=in_channels, relu = False)
        )
        self.downsample = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, padding=(0, 1, 1), stride=2),
            nn.BatchNorm3d(in_channels * 2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, input):
        x = self.conv(input)
        #print(x.size(), input.size())
        #x = self.downsample(input + x)
        x = self.downsample(x)
        return x

class Visit(nn.Module):
    def __init__(self, in_channels = 7):
        super(Visit, self).__init__()
        groups = in_channels // 7

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=groups, out_channels=32, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1, groups=groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1, groups=groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(num_features=64)
        )
        self.layer1 = Residual3(64)
        self.layer2 = Residual3(128)
        self.layer3 = Residual(256)
        self.pool = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3))
        self.relu = nn.LeakyReLU(inplace=True)
        #self.fc = nn.Linear(64, 10)

    def forward(self, input):
        x = self.conv(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), x.size(1), x.size(3), x.size(4))
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.relu(x)

#import pretrainedmodels

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class MultiModalNet(nn.Module):
    def __init__(self, backbone1, drop, pretrained=True, in_channels=14):
        super().__init__()
        # if pretrained:
        #     img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)  # seresnext101
        #     img_model.load_state_dict(torch.load('./weights/se_resnet50.pth'))
        # else:
        #     img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
        import torchvision.models as models
        img_model = models.resnet50()

        self.visit_model = Visit(in_channels=in_channels)

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                        nn.Dropout(drop),
                                        nn.Linear(2048, 256))

        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Linear(2048, 256)
            )

        self.cls = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(320, 9)
        )


    def forward(self, z):
        print(self.visit_model)
        x_img, x_vis = z[0], z[1]
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)
        x_vis = self.visit_model(x_vis)
        x_cat = torch.cat((x_img, x_vis), dim=1)
        x_cat = self.cls(x_cat)
        return x_cat



def build_net(in_channels=14):
    return MultiModalNet(backbone1='se_resnet50', drop=0.5, in_channels=in_channels)


def demo():
    img = torch.rand([4, 3, 100, 100]).cuda()
    visit = torch.rand([4, 7, 26, 24]).cuda()
    visit = visit.unsqueeze(1)
    s = visit.sum(dim=4).unsqueeze(4)
    buf = visit / (s + 1e-6)
    visit = torch.cat((visit, buf), dim = 1)
    net = build_net().cuda()
    c = net((img, visit))
    print(c.size())
demo()
