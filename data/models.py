import torch.nn as nn

from quantizer_torch import quantize


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        ''' declare layers used in this network'''
        # first block
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, img):
        x = self.conv2(self.relu1(self.conv1(img)))

        x = x + img;

        return x

class AnalysisTransformer(nn.Module):

    def __init__(self, args):
        super(AnalysisTransformer, self).__init__()

        ''' declare layers used in this network'''
        # first block
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2)
        self.residualBlock1 = ResidualBlock()
        self.residualBlock2 = ResidualBlock()

        # second block
        self.conv2 = nn.Conv2d(3, 128, kernel_size=3, stride=2)
        self.residualBlock3 = ResidualBlock()
        self.residualBlock4 = ResidualBlock()

        # third block
        self.conv3 = nn.Conv2d(3, 128, kernel_size=3, stride=2)
        self.residualBlock5 = ResidualBlock()
        self.residualBlock6 = ResidualBlock()

        # fourth block
        self.conv4 = nn.Conv2d(3, 128, kernel_size=3, stride=2)
        self.residualBlock7 = ResidualBlock()
        self.residualBlock8 = ResidualBlock()

        #Last layer
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3)

    def forward(self, img):

        x = self.residualBlock2(self.residualBlock1(self.conv1(img)))
        x = self.residualBlock4(self.residualBlock3(self.conv2(x)))
        x = self.residualBlock6(self.residualBlock5(self.conv3(x)))
        x = self.residualBlock8(self.residualBlock7(self.conv4(x)))
        x = self.conv5(x)

        return x

class SynthesisTransformer(nn.Module):

    def __init__(self, args):
        super(SynthesisTransformer, self).__init__()

        ''' declare layers used in this network'''
        #first layer
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3)

        # first block
        self.residualBlock1 = ResidualBlock()
        self.residualBlock2 = ResidualBlock()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(128, 3, kernel_size=3)

        # second block
        self.residualBlock3 = ResidualBlock()
        self.residualBlock4 = ResidualBlock()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(128, 3, kernel_size=3)

        # third block
        self.residualBlock5 = ResidualBlock()
        self.residualBlock6 = ResidualBlock()
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3)

        # fourth block
        self.residualBlock7 = ResidualBlock()
        self.residualBlock8 = ResidualBlock()
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(128, 3, kernel_size=3)

    def forward(self, img):

        x = self.conv1(img)
        x = self.conv2(self.upsample1(self.residualBlock2(self.residualBlock1(x))))
        x = self.conv3(self.upsample2(self.residualBlock4(self.residualBlock3(x))))
        x = self.conv4(self.upsample3(self.residualBlock6(self.residualBlock5(x))))
        x = self.conv5(self.upsample4(self.residualBlock8(self.residualBlock7(x))))

        return x

class CompressionNetwork(nn.Module):

    def __init__(self, args):
        super(CompressionNetwork, self).__init__()

        ''' declare layers used in this network'''

        self.analysis = AnalysisTransformer()
        self.synthesis = SynthesisTransformer()

    def forward(self, img):

        x = self.analysis(img)
        x = quantize(x, (-1,1), 1)
        x = self.synthesis(x)

        return x