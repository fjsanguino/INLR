import numpy as np
import scipy.ndimage
import math
from skimage import io, color
from torch import nn
import torch
from torch.nn import functional as F


# Obtained from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_gaussian_kernel(size, sigma, norm='gauss'):
    if (size % 2) == 0:  # If size is even, size = size - 1
        size -= 1
    x = torch.arange(start=1, end=size + 1, dtype=torch.float32)

    xv, yv = torch.meshgrid(x, x)
    xv, yv = xv.cuda(), yv.cuda()

    kernel = torch.zeros(1, 1, size, size).cuda()
    kernel[0, 0, ...] = torch.exp(-((xv - (size + 1) / 2) ** 2 + (yv - (size + 1) / 2) ** 2) / (2 * sigma ** 2))

    if norm != 'gauss':
        kernel /= kernel.sum()
    else:
        kernel /= 2 * math.pi * sigma ** 2

    return kernel


def INRF_B(u, param):
    L = color.rgb2lab(u) / 100
    L = L[:, :, 0]
    print(np.allclose(L, np.loadtxt('L.txt', delimiter=','), rtol=1e-04, atol=1e-06))

    h = matlab_style_gauss2D((2 * param['sigmaMu'], 2 * param['sigmaMu']), param['sigmaMu'])
    #mUL_javi = scipy.ndimage.correlate(L, h_javi, mode='constant', cval=0, origin=-1)
    #print(np.allclose(mUL_javi, np.loadtxt('mUL.txt', delimiter=','), rtol=1e-04, atol=1e-06)) #tested in matlab with sigmaMU = 5

    L = torch.from_numpy(L).cuda()
    L = torch.reshape(L, (1, 1, L.size()[0], L.size()[1]))
    h = torch.from_numpy(h).cuda()
    h = torch.reshape(h, (1, 1, h.size()[0], h.size()[1]))

    mUL = conv(L, h)
    print(np.allclose(mUL.reshape(mUL.size()[-2], mUL.size()[-1]).cpu().numpy(), np.loadtxt('mUL.txt', delimiter=','), rtol=1e-04,
                      atol=1e-06))  # tested in matlab with sigmaMU = 5 (tamaño filtro par)

    INRF = computeRuInterp_wg_noGPU(L, param['sigmaW'], param['p'], param['q'], param['sigmag'], param['levels_approx'])

    return mUL - param['lambd'] * INRF


def computeRuInterp_wg_noGPU(u, sigmaW, p, q, sigmag, steps):
    if sigmag == 0:
        gu = u
    else:
        # g = matlab_style_gauss2D((2 * sigmag, 2 * sigmag), sigmag)
        # gu = scipy.ndimage.correlate(u, g, mode='reflect', origin=-1)
        # print(np.allclose(gu, np.loadtxt('gu.txt',delimiter=','), rtol=1e-04, atol=1e-06)) #tested in matlab with sigmag = 1

        # todo: test
        g = torch.from_numpy(matlab_style_gauss2D((2 * sigmag, 2 * sigmag), sigmag)).cuda()
        g = torch.reshape(g, (1, 1, g.size()[0], g.size()[1]))
        gu = conv_reflection(u, g)
        print(np.allclose(gu.reshape(gu.size(2),gu.size(3)).cpu().numpy(), np.loadtxt('gu.txt', delimiter=','), rtol=1e-04,
                          atol=1e-06))  # tested in matlab with sigmag = 1, la última columna tiene un error de 0,055
        '''--------------------------------'''

    levels = torch.linspace(torch.min(gu), torch.max(gu), steps).cuda()

    R_L = createPiecewiseR(gu, levels, p, q, sigmaW)
    R_U = interpolate(gu, R_L, levels)

    return R_U


def interpolate(u, R_L, levels):
    L = len(levels)
    R = np.zeros(np.shape(u))
    level_step = levels[1] - levels[0]

    for j in range(0, L - 1):
        indexL = (u >= levels[j]) & (u < levels[j + 1])

        #todo: cambiar si cambia
        R_Lj = R_L[:, :, j]
        R_Lj1 = R_L[:, :, j + 1]

        R[indexL] = R_Lj[indexL] + (R_Lj1[indexL] - R_Lj[indexL]) * (u[indexL] - levels[j]) / level_step

    return R


def createPiecewiseR(u, levels, p, q, sigmaW):
    L = len(levels)
    (M, N) = (u.size(2), u.size(3))
    R_L = torch.zeros(M,N,L).cuda()
    #R_L = torch.from_numpy(np.zeros((M, N, L))).cuda()

    # w = createW(M, N, sigmaW)
    # print(np.allclose(w, np.loadtxt('w.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    w = torch.from_numpy(createW(M, N, sigmaW)).cuda()
    print(np.allclose(w.cpu().numpy(), np.loadtxt('w.txt', delimiter=','), rtol=1e-15,
                      atol=1e-08))  # tested in matlab with M = 512, N = 512 and sigmaW = 25

    # not tested
    # for l in levels:
    # R_L[:,:,l] = convolve(sigmoidatan(u-l), w, sigmaW)
    # print(np.allclose(R_L, np.loadtxt('R_L.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    '''------------------------------------------------'''
    #todo: cambiar para que se pueda meter en conv con img de tamaño (L, 1, h, w) y kernel (1, 1, h',w')
    w = torch.reshape(w, (1, 1, w.size()[0], w.size()[1]))
    for l in range(len(levels)):
        R_L_l = conv(sigmoidatan(u - levels[l]), w)
        R_L[:, :, l] = R_L_l.reshape(R_L_l.size(2), R_L_l.size(3))

    R_L = R_L.reshape(50*512*512).cpu().numpy()
    print(np.allclose(R_L,
                np.loadtxt('../../data/R_L.txt', delimiter=',').reshape((50, 512, 512)), rtol=1e-03,
                atol=1e-06))  # tested in matlab with M = 512, N = 512 and sigmaW = 25
    exit(0)
    return R_L


def convolve(u, w, pad):
    pad_i = pad
    pad_j = pad
    (M, N) = np.shape(u)

    w = np.pad(w, (pad_i, pad_j), 'constant')
    w = np.fft.fftshift(w)
    W = np.fft.fft2(w)

    pad_u = np.pad(u, (pad_i, pad_j), 'constant')
    U = np.fft.fft2(pad_u)

    wu = np.fft.ifft2(W * U)

    return wu[pad_i + 1: pad_i + M, pad_j + 1: pad_j + N]


def conv(img, kernel):
    padding = np.int((kernel.shape[-1]) // 2)
    return F.conv2d(input=img, weight=kernel, padding=padding)[:,:,1:,1:]


def conv_reflection(img, kernel):
    padding = np.int((kernel.shape[-1]) / 2)
    #matlab y scipy.ndimage.correlate reflejan los bordes y nn.ReflectionPad2d de Pytorch no
    pad_rep = nn.ReplicationPad2d(1)
    pad_ref = nn.ReflectionPad2d(padding)
    pad_img = pad_ref(pad_rep(img))

    pad_img_size_x = pad_img.size(2)
    pad_img_size_y = pad_img.size(3)
    pad_img = pad_img[:, :, torch.arange(pad_img_size_x) != padding, : ]
    pad_img = pad_img[:, :, torch.arange(pad_img.size(2)) != pad_img.size(2)-padding, :]
    pad_img = pad_img[:, :, :, torch.arange(pad_img_size_y) != padding ]
    pad_img = pad_img[:, :, :, torch.arange(pad_img.size(3)) != pad_img.size(3)-padding]

    return F.conv2d(input=pad_img, weight=kernel)[:,:,1:,1:]


def createW(M, N, sigmaW):
    J, I = np.ogrid[1:N + 1, 1:M + 1]
    w = np.exp(-((I - M / 2) * (I - M / 2) + (J - N / 2) * (J - N / 2)) / (2. * sigmaW * sigmaW))
    return w / w.sum()


def sigmoidatan(v):
    return torch.atan(10 * v)


if __name__ == "__main__":
    filename = 'Lenna.png'
    param = {
        'sigmaMu': 5,
        'sigmaW': 25,
        'p': 0,
        'q': 0,
        'sigmag': 1,
        'levels_approx': 50,
        'lambd': 3
    }

    rgb = io.imread(filename)
    INRF_B(rgb, param)
