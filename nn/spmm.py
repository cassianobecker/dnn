from torch_scatter import scatter_add
import torch


def spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out.permute(-1, 0) * value
    out = out.permute(-1, 0)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out


def spmm_batch_2(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col]
    try:
        sh = out.shape[2]
    except:
        out = out.unsqueeze(-1)
        sh = 1

    # out = out.permute(1, 2, 0)
    # out = torch.mul(out, value.repeat(-1, sh))
    # out = out.permute(1, 2, 0)
    temp = value.expand(sh, value.shape[0]).permute(1, 0)
    out = torch.einsum("abc,bc->abc", out, temp)
    out = scatter_add(out, row, dim=1, dim_size=m)

    return out


def spmm_batch_3(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col, :]
    try:
        sh = out.shape[3:]
        sh = matrix.shape[2:]
    except:
        out = out.unsqueeze(-1)
        sh = matrix.shape[2:]

    # out = out.permute(1, 2, 0)
    # out = torch.mul(out, value.repeat(-1, sh))
    # out = out.permute(1, 2, 0)
    sh = sh + (value.shape[0],)
    temp = value.expand(sh)
    temp = temp.permute(2, 0, 1)
    out = torch.einsum("abcd,acd->abcd", out, temp)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out
