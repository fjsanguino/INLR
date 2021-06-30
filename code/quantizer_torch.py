import torch

def quantize(x, centers, sigma):
    """ :return qsoft, qhard, symbols """
    return _quantize1d(x, centers, sigma)


def _quantize1d(x, centers, sigma):


    x_shape_BCwh = x.size()
    B = x_shape_BCwh[0]  # B is not necessarily static (batch)
    C = int(x.shape[1])  # C is static (channels)
    x = torch.reshape(x, [B, C, -1])

    # make x into (B, C, m, 1)
    x = x.unsqueeze(-1)

    # dist is (B, C, m, L), contains | x_i - c_j | ^ 2
    dist = torch.square(torch.abs(x - centers)) # realmente sobra torch.abs

    # (B, C, m, L)
    m = torch.nn.Softmax(dim=-1)
    phi_soft = m(-sigma * dist)

    matmul_innerproduct = phi_soft * centers  # (B, C, m, L)
    final = torch.sum(matmul_innerproduct, dim=3)  # (B, C, m)

    return torch.reshape(final, x_shape_BCwh)



