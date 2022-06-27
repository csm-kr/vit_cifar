import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    # [length, d_model]
    return pe


def cosine_simirarity(positinal_embeddings):
    pe = positinal_embeddings  # [length, dim]
    pe = pe.squeeze(0)
    assert len(pe.size()) == 2, 'pe must have 2-dim shape.'

    result = []
    for pe_compoent in pe:
        result.append(nn.functional.cosine_similarity(pe_compoent.unsqueeze(0).expand_as(pe), pe, dim=1))
    return torch.stack(result, dim=0)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


if __name__ == '__main__':
    pe = positionalencoding1d(384, 65)
    print(pe.size())

    pe2 = positionalencoding2d(384, 8, 8)
    pe2 = pe2.view(384, 64).permute(1, 0)
    print(pe2.size())

    cosine_map = cosine_simirarity(pe2)
    plt.figure('cosine_simirarity')
    plt.imshow(cosine_map)
    plt.show()
    #
    # input1 = torch.randn(100, 128)
    # input2 = torch.randn(100, 128)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # output = cos(input1, input2)
    # print(output.size())

