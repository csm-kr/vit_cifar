import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt


from sklearn.utils.extmath import cartesian
import numpy as np


def cdist(a, b):
    differences = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    distances = differences.sqrt()
    # print(distances)
    ret = torch.exp(distances) ** -0.5
    return ret


# class AdvancedWeightedHausdorffDistance(nn.Module):
#     def __init__(self,
#                  resized_height=100,
#                  resized_width=100,
#                  p=-1):
#
#         super().__init__()
#         self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
#                                                              np.arange(resized_width)]))
#         # Convert to appropiate type
#         self.all_img_locations = self.all_img_locations.to(device=device,
#                                                            dtype=torch.get_default_dtype())
#         self.p = p
#
#     def set_init(self, resized_height, resized_width):
#         self.height = resized_height
#         self.width = resized_width
#         self.resized_size = torch.tensor([resized_height,
#                                           resized_width],
#                                          dtype=torch.get_default_dtype(),
#                                          device=device)
#         self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
#         self.n_pixels = resized_height * resized_width
#         self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
#                                                              np.arange(resized_width)]))
#         # Convert to appropiate type
#         self.all_img_locations = self.all_img_locations.to(device=device,
#                                                            dtype=torch.get_default_dtype())
#
#     def map2coord(self, map, thres=1.0):
#         # gt_map : [B, anchors]
#         batch_size = map.size(0)
#         mask_100_ = map.reshape(batch_size, -1)  # [B, 10000]
#         mask_100 = (mask_100_ >= thres).type(torch.float32)  # [0, 1] 로 바꿔버리기
#
#         nozero_100 = []
#         batch_matrices_100 = []
#
#         for b in range(batch_size):
#             nozero_100.append(mask_100[b].nonzero().squeeze())
#             coordinate_matrix_100 = torch.from_numpy(cartesian([np.arange(self.height), np.arange(self.width)]))
#             batch_matrices_100.append(coordinate_matrix_100)
#
#         coordinate_matries_100 = torch.stack(batch_matrices_100, dim=0)
#         mask_100_vis = mask_100.view(-1, self.height, self.width)
#
#         # make seq gt
#         seq_100 = []
#         for b in range(batch_size):
#             seq_100.append(coordinate_matries_100[b][nozero_100[b]].to(device))
#         return seq_100, mask_100_vis
#
#     def forward(self, prob_map, gt_map):
#
#         gt, mask_100_vis = self.map2coord(map=gt_map)
#         orig_sizes = torch.LongTensor([[self.height, self.width], [self.height, self.width]]).to(device)
#         _assert_no_grad(gt)
#
#         assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
#         assert prob_map.size()[1:3] == (self.height, self.width), \
#             'You must configure the WeightedHausdorffDistance with the height and width of the ' \
#             'probability map that you are using, got a probability map of size %s' \
#             % str(prob_map.size())
#
#         batch_size = prob_map.shape[0]
#         assert batch_size == len(gt)
#
#         terms_1 = []
#         terms_2 = []
#         for b in range(batch_size):
#
#             # One by one
#             prob_map_b = prob_map[b, :, :]
#             gt_b = gt[b]
#             orig_size_b = orig_sizes[b, :]
#             norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)
#
#             # Corner case: no GT points
#             if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
#                 terms_1.append(torch.tensor([0],
#                                             dtype=torch.get_default_dtype()))
#                 terms_2.append(torch.tensor([self.max_dist],
#                                             dtype=torch.get_default_dtype()))
#                 continue
#
#             # Pairwise distances between all possible locations and the GTed locations
#             n_gt_pts = gt_b.size()[0]
#             normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
#             normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
#             d_matrix = cdist(normalized_x, normalized_y)


def positional_embedding(d_model, length=64):
    all_img_locations = torch.from_numpy(cartesian([np.arange(length ** 0.5),
                                                    np.arange(length ** 0.5)]))

    dist = []
    for coords in all_img_locations:
        for coords_i in all_img_locations:
            dist.append(cdist(coords, coords_i))

            # print("1/exp(d) :", cdist(coords, coords_i))
    dist_map = torch.stack(dist, dim=0).view(64, 64)

    plt.figure('dist_map')
    plt.imshow(dist_map)
    plt.show()

    #
    # print(all_img_locations.shape)
    # normalized_x = all_img_locations[:, 0].repeat(length, 1)
    # normalized_y = all_img_locations[:, 1].repeat(length, 1)
    # # normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
    # cdists = cdist(normalized_x, normalized_y)
    # print(cdists.size())

    return dist_map.unsqueeze(-1)



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

    positional_embedding(d_model=384, length=64)

    # pe = positionalencoding1d(384, 65)
    # print(pe.size())
    #
    # pe2 = positionalencoding2d(384, 8, 8)
    # pe2 = pe2.view(384, 64).permute(1, 0)
    # print(pe2.size())
    #
    # cosine_map = cosine_simirarity(pe2)
    # plt.figure('cosine_simirarity')
    # plt.imshow(cosine_map)
    # plt.show()
    #
    # input1 = torch.randn(100, 128)
    # input2 = torch.randn(100, 128)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # output = cos(input1, input2)
    # print(output.size())

