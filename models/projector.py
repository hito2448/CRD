import torch
import torch.nn as nn

class Projector(torch.nn.Module):
    def __init__(self):
        super(Projector, self).__init__()

        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])

        return [proj_a, proj_b, proj_c]


class Projector_Amply_upsample(torch.nn.Module):
    def __init__(self):
        super(Projector_Amply_upsample, self).__init__()

        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

        )
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])
        return [proj_a, proj_b, proj_c]


class NormalProjector_8(torch.nn.Module):
    def __init__(self):
        super(NormalProjector_8, self).__init__()

        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])

        return [proj_a, proj_b, proj_c], [proj_a, proj_b, proj_c]

