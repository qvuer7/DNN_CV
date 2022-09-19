from torch import nn
import torch

'''
Resnet class realisied for classification task solution
for segmentation change head to: 
    nn.Sequential(
        nn.Conv2d(2048, 512, 3, 1, 1), 
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Conv2d(512, n_classes)

'''


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, stride = 1, downsample = None):
        super(Block, self).__init__()
        if n_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()

        if self.expansion == 4:
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size = 3, stride = stride, padding = 1)
        else:
            self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels * self.expansion, kernel_size = 1, stride = 1, padding =0 )
        self.bn3 = nn.BatchNorm2d(self.out_channels*self.expansion)
        self.relu3 = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        if self.expansion == 4:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)


        x = self.conv3(x)
        x = self.bn3(x)



        if self.downsample is not None:
            residual = self.downsample(residual)


        x+= residual
        x = self.relu3(x)

        return x


class resnet(nn.Module):
    def __init__(self,n_layers, block, num_classes):
        super(resnet, self).__init__()
        self.n_layers = n_layers

        if self.n_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if self.n_layers == 18:
            layers = [2, 2, 2, 2]
        elif self.n_layers == 34 or self.n_layers == 50:
            layers = [3, 4, 6, 3]
        elif self.n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.mpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self.create_layer(stride = 1, block = block, n_blocks = layers[0], intermidiate_channels = 64)
        self.layer2 = self.create_layer(stride = 2, block = block, n_blocks = layers[1], intermidiate_channels = 128)
        self.layer3 = self.create_layer(stride = 2, block = block, n_blocks = layers[2], intermidiate_channels = 256)
        self.layer4 = self.create_layer(stride = 2, block = block, n_blocks = layers[3], intermidiate_channels = 512)

        self.apool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)


    def create_layer(self, stride, block, n_blocks, intermidiate_channels):
        layers = []

        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermidiate_channels* self.expansion, kernel_size = 1, stride = stride, padding = 0),
            nn.BatchNorm2d(intermidiate_channels* self.expansion)
        )
        layers.append(block(self.in_channels, intermidiate_channels, self.n_layers, stride, downsample))
        self.in_channels = intermidiate_channels * self.expansion
        for i in range(n_blocks - 1):
            layers.append(block(self.in_channels, intermidiate_channels, self.n_layers))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #classification head
        x = self.apool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x



def test():
    resnet50 = resnet(block = Block , n_layers = 50, num_classes = 10)
    x = torch.randn((1,3,320,320))

    out = resnet50(x)
    print(x.shape)
    print(out.shape)

if __name__ == '__main__':
    test()



