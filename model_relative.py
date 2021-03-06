import math
import torch
import torch.nn as nn

from sklearn.utils.extmath import cartesian
import numpy as np


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


def cdist(a, b):
    differences = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    distances = differences.sqrt()
    # print(distances)
    ret = torch.exp(distances) ** -0.33 + 1
    return ret


def positional_embedding(length=64):
    all_img_locations = torch.from_numpy(cartesian([np.arange(length ** 0.5),
                                                    np.arange(length ** 0.5)]))

    dist = []
    for coords in all_img_locations:
        for coords_i in all_img_locations:
            dist.append(cdist(coords, coords_i))
    dist_map = torch.stack(dist, dim=0).view(length, length)  # [64, 64]
    dist_map = dist_map.type(torch.float32)
    # dist_map = dist_map.expand([1, length, d_model])

    return dist_map


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
    pe = pe.view(d_model * 2, width * height).permute(1, 0)
    return pe


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, patch_size, image_size, dropout_ratio=0.1, is_cls_token=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size                                                          # 16 x 16 in ViT
        self.num_patches = (image_size // patch_size) ** 2                                    # L
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if is_cls_token else None       # [1, 1, D]
        self.patch_embedding_projection = nn.Conv2d(3, dim, patch_size, stride=patch_size)    # Non overlap projection

        # # # sinusoid PE 1d
        # self.position_embedding = positionalencoding1d(dim, self.num_patches + 1).unsqueeze(0) if is_cls_token \
        #     else positionalencoding1d(dim, self.num_patches).unsqueeze(0)  # [1, N (+1), D]

        # # sinusoid PE 2d
        # self.position_embedding = positionalencoding2d(dim, self.num_patches + 1, self.num_patches + 1).unsqueeze(0) if is_cls_token \
        #     else positionalencoding2d(dim, (image_size // patch_size), (image_size // patch_size)).unsqueeze(0)  # [1, N (+1), D]

        # # sinusoid PE distance
        # self.position_embedding = positional_embedding(length=self.num_patches + 1) if is_cls_token \
        #     else positional_embedding(length=self.num_patches)  # [1, N (+1)]

        # learnable PE
        self.position_embedding = nn.Parameter(torch.empty(1, (self.num_patches + 1), dim)) if is_cls_token \
            else nn.Parameter(torch.empty(1, self.num_patches, dim))                          # [1, N (+1), D]
        torch.nn.init.normal_(self.position_embedding, std=.02)

        self.pos_dropout = nn.Dropout(dropout_ratio)
        self.is_cls_token = is_cls_token

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding_projection(x)                                                # [B, 64, 28, 28]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_patches, self.dim)           # [B, L, D]
        if self.is_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)                            # Expand to [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1)                                             # [B, 1 + N, D]
            # cls_token [cls_token; x_p^1E; x_p^2E; ..., ;x_p^N]
        # else [x_p^1E; x_p^2E; ..., ;x_p^N]
        device = x.get_device()
        self.position_embedding = self.position_embedding.to(device)

        # x = x.permute(0, 2, 1) @ self.position_embedding  # [B, D, L]
        # x = x.permute(0, 2, 1)                            # [B, D, L]

        x += self.position_embedding
        x = self.pos_dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5                          # 1/sqrt(dim)
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(int(self.dim / self.num_heads), self.max_relative_position)
        self.relative_position_v = RelativePosition(int(self.dim / self.num_heads), self.max_relative_position)

        self.q = nn.Linear(dim, dim, bias=True)           # Wq
        self.k = nn.Linear(dim, dim, bias=True)           # Wk
        self.v = nn.Linear(dim, dim, bias=True)           # Wv
        self.o = nn.Linear(dim, dim)                      # Wo

        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.multi_head_dropout = nn.Dropout(dropout_ratio)

        self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.zeros_(self.q.bias)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.zeros_(self.k.bias)
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.zeros_(self.v.bias)
        torch.nn.init.xavier_uniform_(self.o.weight)
        torch.nn.init.zeros_(self.o.bias)

    def forward(self, x):
        assert self.dim % self.num_heads == 0, 'dim ??? ????????? head ??? ????????? ???????????? ??????. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)

        b, l, _, h = *x.shape, self.num_heads                          # b: batch, l : length, h : head
        h_d = int(self.dim / self.num_heads)

        q_ = self.q(x)
        k_ = self.k(x)
        v_ = self.v(x)

        r_q1 = q = q_.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D
        r_k1 = k = k_.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D
        r_v1 = v = v_.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D

        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = q_.permute(1, 0, 2).contiguous().view(l, b * self.num_heads, h_d)
        r_k2 = self.relative_position_k(l, l)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(b, self.num_heads, l, l)
        attn = (attn1 + attn2) / self.scale

        attn = self.attn_dropout(torch.softmax(attn, dim=-1))

        #attn = [batch size, n heads, query len, key len]
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(l, l)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(l, b * self.num_heads, l)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(b, self.num_heads, l, h_d)
        x = weight1 + weight2

        # dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale      # B, H, L, L
        # attn = dots.softmax(dim=-1)
        # x = torch.einsum('bhij,bhjd->bhid', attn, v)                   # B, H, L, H_D
        # x = self.attn_dropout(x)

        x = x.permute(0, 2, 1, 3).reshape(b, l, h * h_d)               # B, L, D
        x = self.o(x)
        x = self.multi_head_dropout(x)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.msa = MultiHeadAttention(dim, num_heads, dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_ratio),
        )
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x_, attn = self.msa(self.layer_norm1(x))
        x = x_ + x                                   # for residual
        x = self.mlp(self.layer_norm2(x)) + x
        return x, attn


class ViT(nn.Module):
    def __init__(self,
                 dim, mlp_dim, num_heads, num_layers,
                 patch_size, image_size, is_cls_token,
                 dropout_ratio, num_classes):
        super().__init__()
        self.is_cls_token = is_cls_token
        self.embedding = EmbeddingLayer(dim, patch_size, image_size, dropout_ratio, is_cls_token)
        self.encoder_list = [EncoderLayer(dim, mlp_dim, num_heads, dropout_ratio) for _ in range(num_layers)]
        self.transformer_module_list = nn.ModuleList(self.encoder_list)
        # self.transformer = nn.Sequential(*self.encoder_list)
        # self.encoder_norm = nn.LayerNorm(dim)

        # init fc
        self.fc = nn.Linear(dim, num_classes)
        self.fc.apply(self.init_weights)

        print("num_params : ", self.count_parameters())

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.bias, std=1e-6)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, is_attn=False):
        batch_size = x.size(0)
        attns = []
        x = self.embedding(x)                                       # [B, L, D]
        for module in self.transformer_module_list:
            x, attn = module(x)
            attns.append(attn)

        # x = self.transformer(x)                                     # [B, L, D]
        if self.is_cls_token:
            x = x[:, 0]                                             # [B, D]
        else:
            x = torch.mean(x, dim=1)                                # [B, D]
        # no encoder norm is better
        # x = self.encoder_norm(x)                                  # [B, D]
        x = self.fc(x)                                              # [B, num_classes]

        if is_attn:
            attn_mask = torch.stack(attns).permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, 17, 17).mean(1)  # [2, 16, 16]
            return x, attn_mask
        return x


if __name__ == '__main__':
    x = torch.randn([2, 3, 32, 32]).cuda()
    vit = ViT(dim=384, mlp_dim=384, num_heads=12, num_layers=7,
              patch_size=4, image_size=32, is_cls_token=False,
              dropout_ratio=0.1, num_classes=10).cuda()

    x = vit(x, False)
    print(x.size())
    # print(attn_mask.size())
    #
    # attn_mask[0] /= attn_mask[0].max()
    # attn_mask_numpy_batch_0 = attn_mask[0].detach().cpu().numpy()
    # import cv2
    # attn_mask_numpy_batch_0 = cv2.resize(attn_mask_numpy_batch_0, (100, 100))
    # cv2.imshow('input', attn_mask_numpy_batch_0)
    # cv2.waitKey()

