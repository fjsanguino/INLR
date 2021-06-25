from .MS_SSIM import MS_SSIM
from .VIF import VIF
from .CW_SSIM import CW_SSIM
from .MAD import MAD
from .FSIM import FSIM
from .GMSD import GMSD
from .VSI import VSI
from .NLPD import NLPD
from .LPIPSvgg import LPIPSvgg
from .DISTS import DISTS

IQA_models_array = ['MS-SSIM', 'VIF', 'CW-SSIM', 'MAD', 'FSIM', 'GMSD', 'VSI', 'NLPD', 'LPIPS', 'DISTS']

def IQA_models_class(model_name):
    if model_name == 'MS-SSIM':
        return MS_SSIM()
    if model_name == 'VIF':
        return VIF()
    if model_name == 'CW-SSIM':
        return CW_SSIM()
    if model_name == 'MAD':
        return MAD()
    if model_name == 'FSIM':
        return FSIM()
    if model_name == 'GMSD':
        return GMSD()
    if model_name == 'VSI':
        return VSI()
    if model_name == 'NLPD':
        return NLPD()
    if model_name == 'LPIPS':
        return LPIPSvgg()
    if model_name == 'DISTS':
        return DISTS()


