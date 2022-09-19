from torch import nn
import torch
from torchvision.models.vgg import vgg16





class vgg(nn.Module):
    def __init__(self, n_layers):
        super(vgg, self).__init__()
        assert n_layers in [11, 13, 16, 19] , f'VGG{n_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 11, 13, 16, 19 '
        self.n_layers = n_layers
        if self.n_layers == 11:
            layers = [1,1,2,2,2]
        elif self.n_layers == 13:
            layers = [2, 2, 2, 2,2]
        elif self.n_layers == 16:
            layers = [2, 2, 3, 3, 3]
        elif self.n_layers == 19:
            layers = [2, 2, 4, 4, 4]



        self.layer0 = self.create_layer(3, 64, layers[0])
        self.layer1 = self.create_layer(64, 128, layers[1])
        self.layer2 = self.create_layer(128, 256, layers[2])
        self.layer3 = self.create_layer(256, 512, layers[3])
        self.layer4 = self.create_layer(512, 512, layers[4])
        self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0,dilation=1, ceil_mode=False)


    def forward(self, x):
        x = self.layer0(x)
        x = self.mp(x)
        x = self.layer1(x)
        x = self.mp(x)
        x = self.layer2(x)
        x = self.mp(x)
        x = self.layer3(x)
        x = self.mp(x)
        x = self.layer4(x)

        return x

    def create_layer(self, in_channels, out_channels, n_blocks):
        layers = []
        conv1 =     nn.Sequential(
                nn.Conv2d(in_channels , out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                nn.ReLU(inplace = True)
            )

        layers.append(conv1)

        for i in range(n_blocks - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                nn.ReLU()))

        return nn.Sequential(*layers)


class vgg_clasifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super(vgg_clasifier, self).__init__()
        self.backbone = backbone
        self.clasifier_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features = 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5, inplace = False),
            nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5, inplace = False),
            nn.Linear(in_features = 4096, out_features = n_classes, bias = True),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.clasifier_head(x)
        return x


def test():
    im = torch.randn((1,3,224, 224))

    v16 = vgg(16)


    vgg_16 = vgg_clasifier(backbone = v16, n_classes = 10)
    out = vgg_16(im)
    print(out)

if __name__ == '__main__':
    test()

