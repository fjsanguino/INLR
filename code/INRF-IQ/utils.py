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
    L = torch.from_numpy(L[:, :, 0]).float().cuda()
    print(np.allclose(L.cpu().numpy(), np.loadtxt('L.txt', delimiter=','), rtol=1e-04, atol=1e-06))

    h_javi = torch.from_numpy(matlab_style_gauss2D((2 * param['sigmaMu'], 2 * param['sigmaMu']), param['sigmaMu'])).cuda()
    mUL_javi = scipy.ndimage.correlate(L, h_javi, mode='constant', cval=0, origin=-1)
    print(np.allclose(mUL_javi, np.loadtxt('mUL.txt',delimiter=','), rtol=1e-04, atol=1e-06)) #tested in matlab with sigmaMU = 5

    # TODO: not tested
    L = torch.reshape(L, (1, 1, L.size()[0], L.size()[1]))
    h = get_gaussian_kernel(2 * param['sigmaMu'], param['sigmaMu'])
    mUL = conv(L, h)
    print(np.allclose(mUL.reshape(L.size()[-2], L.size()[-1]).cpu().numpy(), np.loadtxt('mUL.txt', delimiter=','), rtol=1e-04,
                      atol=1e-06))  # tested in matlab with sigmaMU = 5

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
        gu = conv_reflection(u, g)
        print(np.allclose(gu.cpu().numpy(), np.loadtxt('gu.txt', delimiter=','), rtol=1e-04,
                          atol=1e-06))  # tested in matlab with sigmag = 1
        '''--------------------------------'''

    levels = torch.from_numpy(np.linspace(np.amin(gu), np.amax(gu), steps)).cuda()

    R_L = createPiecewiseR(gu, levels, p, q, sigmaW)
    R_U = interpolate(gu, R_L, levels)

    return R_U


def interpolate(u, R_L, levels):
    L = len(levels)
    R = np.zeros(np.shape(u))
    level_step = levels[1] - levels[0]

    for j in range(0, L - 1):
        indexL = (u >= levels[j]) & (u < levels[j + 1])

        R_Lj = R_L[:, :, j]
        R_Lj1 = R_L[:, :, j + 1]

        R[indexL] = R_Lj[indexL] + (R_Lj1[indexL] - R_Lj[indexL]) * (u[indexL] - levels[j]) / level_step

    return R


def createPiecewiseR(u, levels, p, q, sigmaW):
    L = len(levels)
    (M, N) = np.shape(u)
    R_L = torch.from_numpy(np.zeros((M, N, L))).cuda()

    # w = createW(M, N, sigmaW)
    # print(np.allclose(w, np.loadtxt('w.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    # TODO:test
    w = torch.from_numpy(createW(M, N, sigmaW)).cuda()
    print(np.allclose(w.cpu().numpy(), np.loadtxt('w.txt', delimiter=','), rtol=1e-15,
                      atol=1e-08))  # tested in matlab with M = 512, N = 512 and sigmaW = 25

    # not tested
    # for l in levels:
    # R_L[:,:,l] = convolve(sigmoidatan(u-l), w, sigmaW)
    # print(np.allclose(R_L, np.loadtxt('R_L.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    '''------------------------------------------------'''
    for l in levels:
        R_L[:, :, l] = conv(sigmoidatan(u - l), w)
    print(np.allclose(R_L, np.loadtxt('R_L.txt', delimiter=','), rtol=1e-15,
                      atol=1e-08))  # tested in matlab with M = 512, N = 512 and sigmaW = 25
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
    padding = np.int((kernel.shape[-1] - 1) / 2)
    return F.conv2d(input=img, weight=kernel, padding=padding)


def conv_reflection(img, kernel):
    padding = np.int((kernel.shape[-1] - 1) / 2)
    pad_ref = nn.ReflectionPad2d(padding)
    return F.conv2d(input=pad_ref(img), weight=kernel)


def createW(M, N, sigmaW):
    J, I = np.ogrid[1:N + 1, 1:M + 1]
    w = np.exp(-((I - M / 2) * (I - M / 2) + (J - N / 2) * (J - N / 2)) / (2. * sigmaW * sigmaW))
    return w / w.sum()


def sigmoidatan(v):
    return np.arctan(10 * v)


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
