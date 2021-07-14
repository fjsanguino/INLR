import torch.nn as nn
import torch
import numpy as np

from quantizer_torch import quantize

##########################################
# Implementation of INRF layer on Pytorch
#
##########################################
# Inputs:
#   -dimG: dimension kernel g, not implemented yet, ignore
#   -dimW: dimension of kernel w
#   -numChanIn: dimension of input channels
#   -numChanOut: dimension of output channels
#   -sigma: Nonlinearity used:
#        1: Relu
#        2: Tanh
#        3: pq sigmoid with p=0.7 and q=0.3
#        4: pq sigmoid with p and q trainable
#        5: Sinusoid activation function
#   -paramSigma: Normalization factor, initial weight values will be divided by this
#   -lambdaV: weight of the nonlinearity part
#   -stride: not implemented yet, ignore

class INRF(nn.Module):

    def __init__(self, dimG=1, dimW=3, numChanIn=3, numChanOut=3, sigma= 1, paramSigma= 12.0, lambdaV = 2.0, stride=1):

        super(INRF, self).__init__()
        self.g = torch.nn.Conv2d(numChanIn, numChanOut, dimG, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.w = nn.Parameter(torch.randn(dimW*dimW, numChanOut, numChanIn, 1, 1,requires_grad=True)/paramSigma)
        self.stride = stride
        self.lamda = lambdaV
        self.pandQ = nn.Parameter(torch.autograd.Variable(torch.from_numpy(np.asarray([0.5,0.5])), requires_grad=True).type(torch.FloatTensor))
        self.paramSigma = paramSigma
        self.numChanIn = numChanIn
        self.dimW = dimW
        # Create difference masks
        self.diffMasks = nn.Parameter(self.createMask(dimW, numChanIn), requires_grad=False).cpu()
        #self.diffMasks = nn.Parameter(self.createMask(dimW, numChanIn), requires_grad=False).cuda()
        self.sigma = sigma

    def forward(self, x):

        for i in range(self.dimW * self.dimW):

            pad = self.dimW // 2
            # Compute differences
            convw = nn.functional.conv2d(x, self.diffMasks[i, :, :, :].unsqueeze(1),
                                                padding=(pad, pad),
                                                groups=self.numChanIn)

            # Non-linearity
            sub = convw +  self.lamda*self.activation(x, convw, self.sigma)

            if i == 0:
                w = nn.functional.conv2d(sub, self.w[i, :, :, :], stride=self.stride)
            else:
                w += nn.functional.conv2d(sub, self.w[i, :, :, :], stride=self.stride)

        return w

    def activation(self, x, convx, option):

        if(option==1):
            return nn.functional.relu(x-convx)

        elif(option==2):
            return nn.functional.tanh(x-convx)

        elif (option == 3):

            signs = torch.sign(x - convx)
            signsNegative = torch.sign(x - convx)*(-1)
            diff = nn.functional.relu(x - convx)
            diff2 = torch.nn.functional.relu((x - convx)*signsNegative)
            positivePart = diff**0.8
            negativePart = diff2 ** 0.3
            out = (positivePart +  negativePart)*signs

            return out

        elif (option == 4):

            signs = torch.sign(x - convx)
            signsNegative = torch.sign(x - convx) * (-1)
            diff = torch.nn.functional.relu(x - convx)
            diff2 = torch.nn.functional.relu((x - convx) * signsNegative)
            positivePart = diff ** self.pandQ[0]
            negativePart = diff2 ** self.pandQ[1]
            out = (positivePart + negativePart) * signs

            return out

        elif (option == 5):

            return torch.sin(x - convx)


    def createMask(self, dimW, numChanIn):

        n = numChanIn
        # Create empty zero matrix
        diffMasksVect = np.zeros((dimW * dimW, numChanIn, dimW, dimW))
        # Counters for rows and cols
        contRow = 0
        contCols = 0

        for i in range((dimW * dimW)):

            w = np.zeros((dimW,dimW))
            w[contRow,contCols] = 1
            contCols += 1
            if ( contCols>=dimW):
                contRow += 1
                contCols = 0

            diffMasksVect[i, :, :, :] = np.stack([w] * n, axis=0)

        return torch.Tensor(diffMasksVect)


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        ''' declare layers used in this network'''
        # first block
        self.inrf1 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)
        self.relu1 = nn.ReLU()
        self.inrf2 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)

    def forward(self, img):
        x = self.inrf2(self.relu1(self.inrf1(img)))

        x = x + img

        return x

class AnalysisTransformer(nn.Module):

    def __init__(self):
        super(AnalysisTransformer, self).__init__()

        ''' declare layers used in this network'''
        # first block
        self.inrf1 = INRF(dimG=1, dimW=5,numChanIn=3, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0, stride=2)
        self.residualBlock1 = ResidualBlock()
        self.residualBlock2 = ResidualBlock()

        # second block
        self.inrf2 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0, stride=2)
        self.residualBlock3 = ResidualBlock()
        self.residualBlock4 = ResidualBlock()

        # third block
        self.inrf3 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0, stride=2)
        self.residualBlock5 = ResidualBlock()
        self.residualBlock6 = ResidualBlock()

        # fourth block
        self.inrf4 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0, stride=2)
        self.residualBlock7 = ResidualBlock()
        self.residualBlock8 = ResidualBlock()

        #Last layer
        self.inrf5 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=64, sigma=1, paramSigma=12, lambdaV=2.0)

    def forward(self, img):
        #print(img.shape)
        x = self.residualBlock2(self.residualBlock1(self.inrf1(img)))
        #print(x.shape)
        x = self.residualBlock4(self.residualBlock3(self.inrf2(x)))
        #print(x.shape)
        x = self.residualBlock6(self.residualBlock5(self.inrf3(x)))
        #print(x.shape)
        x = self.residualBlock8(self.residualBlock7(self.inrf4(x)))
        #print(x.shape)
        x = self.inrf5(x)

        return x

class SynthesisTransformer(nn.Module):

    def __init__(self):
        super(SynthesisTransformer, self).__init__()

        ''' declare layers used in this network'''
        #first layer
        self.inrf1 = INRF(dimG=1, dimW=5,numChanIn=64, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)

        # first block
        self.residualBlock1 = ResidualBlock()
        self.residualBlock2 = ResidualBlock()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.inrf2 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)

        # second block
        self.residualBlock3 = ResidualBlock()
        self.residualBlock4 = ResidualBlock()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.inrf3 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)

        # third block
        self.residualBlock5 = ResidualBlock()
        self.residualBlock6 = ResidualBlock()
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.inrf4 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=128, sigma=1, paramSigma=12, lambdaV=2.0)

        # fourth block
        self.residualBlock7 = ResidualBlock()
        self.residualBlock8 = ResidualBlock()
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.inrf5 = INRF(dimG=1, dimW=5,numChanIn=128, numChanOut=3, sigma=1, paramSigma=12, lambdaV=2.0)

    def forward(self, img):

        x = self.inrf1(img)
        #print(x.shape)
        x = self.inrf2(self.upsample1(self.residualBlock2(self.residualBlock1(x))))
        #print(x.shape)
        x = self.inrf3(self.upsample2(self.residualBlock4(self.residualBlock3(x))))
        #print(x.shape)
        x = self.inrf4(self.upsample3(self.residualBlock6(self.residualBlock5(x))))
        #print(x.shape)
        x = self.inrf5(self.upsample4(self.residualBlock8(self.residualBlock7(x))))
        #print(x.shape)

        return x

class CompressionNetwork(nn.Module):

    def __init__(self):
        super(CompressionNetwork, self).__init__()

        ''' declare layers used in this network'''

        self.analysis = AnalysisTransformer()
        self.synthesis = SynthesisTransformer()

    def forward(self, img):

        x = self.analysis(img)
        #print(x.shape)
        x = quantize(x, torch.from_numpy(np.asarray((-1, 1))).cuda(), 1)
        #print(x.shape)
        x = self.synthesis(x)

        return x