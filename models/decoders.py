from .resnet import bn, bn_fuse
from .de_resnet import de_wide_resnet50_2, de_wide_resnet50_2_skip
from .projector import *

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ResNet50Decoder(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50Decoder, self).__init__()
        self.bn = bn()
        self.decoder = de_wide_resnet50_2(pretrained=pretrained)

    def forward(self, x):
        x = self.bn(x)
        x = self.decoder(x)

        return x


class ResNet50DualModalDecoder(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50DualModalDecoder, self).__init__()
        self.bn = bn_fuse()
        self.decoder = de_wide_resnet50_2_skip(pretrained=pretrained)
        # self.projector_filter = NormalProjector()
        self.projector_filter = NormalProjector_8()
        self.projector_amply = Projector_Amply_upsample()

    def forward(self, x, x_assist, attn=False, noise=False):
        x_encoder = x
        proj_filter, x_assist_proj = self.projector_filter(x_assist)
        proj_amply = self.projector_amply(x_assist)

        x_bn = self.bn(x, proj_filter)

        x_amply, x = self.decoder.forward_fuse(x_bn, proj_amply)

        return proj_filter, proj_amply, x, x_amply

