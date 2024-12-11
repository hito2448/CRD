import torch
from .resnet import wide_resnet50_2

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ResNet50Encoder(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        self.encoder = wide_resnet50_2(pretrained=pretrained)

    def forward(self, x):
        # with torch.no_grad():
        #     x = self.encoder(x)
        x = self.encoder(x)

        return x

