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

def matlab_style_gauss2D_pytorch(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(start=-m, end=m+1, dtype=torch.float32), torch.arange(start=-n, end=n+1, dtype=torch.float32))
    h = torch.zeros(1, 1, shape[0], shape[1], dtype=torch.float64).cuda()
    h[0, 0, ...] = torch.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def createW(M, N, sigmaW):
    J, I = np.ogrid[1:N + 1, 1:M + 1]
    w = np.exp(-((I - M / 2) * (I - M / 2) + (J - N / 2) * (J - N / 2)) / (2. * sigmaW * sigmaW))
    return w / w.sum()

def createW_pytorch(M, N, sigmaW):
    J, I  = torch.meshgrid(torch.arange(start=1, end=N+1, dtype=torch.float64), torch.arange(start=1, end=M+1, dtype=torch.float64))
    w = torch.zeros(1, 1, M, N, dtype=torch.float64).cuda()
    w[0, 0, ...] = torch.exp(-((I - M / 2) * (I - M / 2) + (J - N / 2) * (J - N / 2)) / (2. * sigmaW * sigmaW))
    return w / w.sum()


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

    h = matlab_style_gauss2D_pytorch((2 * param['sigmaMu'], 2 * param['sigmaMu']), param['sigmaMu'])
    #mUL_javi = scipy.ndimage.correlate(L, h_javi, mode='constant', cval=0, origin=-1)
    #print(np.allclose(mUL_javi, np.loadtxt('mUL.txt', delimiter=','), rtol=1e-04, atol=1e-06)) #tested in matlab with sigmaMU = 5

    L = torch.from_numpy(L).double().cuda()
    L = torch.reshape(L, (1, 1, L.size()[0], L.size()[1]))

    mUL = conv(L, h)
    print(np.allclose(mUL.reshape(mUL.size()[-2], mUL.size()[-1]).cpu().numpy(), np.loadtxt('mUL.txt', delimiter=','), rtol=1e-04, atol=1e-06))  # tested in matlab with sigmaMU = 5 (tamaño filtro par)

    INRF = computeRuInterp_wg_noGPU(L, param['sigmaW'], param['p'], param['q'], param['sigmag'], param['levels_approx'])

    return mUL - param['lambd'] * INRF


def computeRuInterp_wg_noGPU(u, sigmaW, p, q, sigmag, steps):
    if sigmag == 0:
        gu = u
    else:
        # g = matlab_style_gauss2D((2 * sigmag, 2 * sigmag), sigmag)
        # gu = scipy.ndimage.correlate(u, g, mode='reflect', origin=-1)
        # print(np.allclose(gu, np.loadtxt('gu.txt',delimiter=','), rtol=1e-04, atol=1e-06)) #tested in matlab with sigmag = 1

        g = matlab_style_gauss2D_pytorch((2 * sigmag, 2 * sigmag), sigmag)
        gu = conv_reflection(u, g)
        #print(np.allclose(gu.reshape(gu.size(2),gu.size(3)).cpu().numpy(), np.loadtxt('gu.txt', delimiter=','), rtol=1e-04, atol=1e-06))  # tested in matlab with sigmag = 1, la última columna tiene un error de 0,055
        '''--------------------------------'''

    levels = torch.linspace(torch.min(gu), torch.max(gu), steps, dtype=torch.float64).cuda()

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
    nlevels = len(levels)
    (M, N) = (u.size(2), u.size(3))
    #R_L = torch.zeros(M,N,nlevels).cuda()
    N, C, rows, cols = u.shape
    r_levels = torch.zeros(size=((N, C, rows, cols, nlevels)), dtype=torch.float64).cuda()
    #R_L = torch.from_numpy(np.zeros((M, N, L))).cuda()

    # w = createW(M, N, sigmaW)
    # print(np.allclose(w, np.loadtxt('w.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    w = createW_pytorch(rows, cols, sigmaW)
    #print(np.allclose(w.cpu().numpy(), np.loadtxt('w.txt', delimiter=','), rtol=1e-15, atol=1e-08))  # tested in matlab with M = 512, N = 512 and sigmaW = 25

    # not tested
    # for l in levels:
    # R_L[:,:,l] = convolve(sigmoidatan(u-l), w, sigmaW)
    # print(np.allclose(R_L, np.loadtxt('R_L.txt',delimiter=','), rtol=1e-15, atol=1e-08)) #tested in matlab with M = 512, N = 512 and sigmaW = 25

    '''------------------------------------------------'''
    #todo: cambiar para que se pueda meter en conv con img de tamaño (L, 1, h, w) y kernel (1, 1, h',w')
    #for l in range(len(levels)):
    #    R_L_l = conv(sigmoidatan(u - levels[l]), w)
    #    R_L[:, :, l] = R_L_l.reshape(R_L_l.size(2), R_L_l.size(3))
    for nlevel in range(nlevels):
        r_levels[..., nlevel] = conv(img=sigmoidatan(levels[nlevel] - u),
                                              kernel=w)
    print("R_L calculated")
    #R_L = R_L.cpu().numpy()
    print(np.allclose(r_levels.cpu().numpy().reshape(512, 512, 2),
                np.loadtxt('../../data/R_L_2.txt', delimiter=',').reshape((512, 512, 2)), rtol=1e-03,
                atol=1e-06))  # tested in matlab with M = 512, N = 512 and sigmaW = 25, nlevels = 2
    exit(0)
    return r_levels

#r_levels_matlab sum = 13331.223509999865
#r_levels sum = 11426.77578896794

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
        'levels_approx': 2,
        'lambd': 3
    }

    rgb = io.imread(filename)
    INRF_B(rgb, param)
