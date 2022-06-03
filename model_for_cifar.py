import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


#########################################################################
# https://github.com/tahmid0007/VisionTransformer/blob/main/Google_ViT.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, mlp_size, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, mlp_size)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(mlp_size, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=12, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  # b: batch, n : length, h : head
        qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv m m x3   [2, 14 * 14 + 1, 768 * 3]
        # q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions
        assert self.dim % self.heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.heads, self.dim % self.heads)
        h_d = int(self.dim / self.heads)

        qkv_ = qkv.permute(2, 0, 1).view(3, h, h_d, b, n).permute(0, 3, 1, 4, 2)
        q, k, v = qkv_[0], qkv_[1], qkv_[2]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # dots = torch.einsum('bhid,bhjd->bhij', q, k)

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        # attn = torch.sigmoid(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * h_d)
        # out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, layers, heads, mlp_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_size, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for layer_num, (attention, mlp) in enumerate(self.layers):
            x = attention(x, mask=mask)  # go to attention             # [B, 257, 256]
            x = mlp(x)  # go to MLP_Block
            # print(str(layer_num) + str(x.size()))                                            # [B, 257, 256]
        return x

#########################################################################

# image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0.1, emb_dropout = 0.1
class VisionTransformer(nn.Module):
    def __init__(self, image_size, dim, patch_size, layers=7, heads=12, mlp_size=384, num_classes=10, dropout=0.0):
        '''

        :param dim: 어느 dim 으로 임베딩 하느냐 (D)
        :param patch_size: 이미지를 자르는 하나의 patch
        '''
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size                                                              # 1 // 4
        self.num_patches = (image_size // patch_size) ** 2                                        # 784
        # number of patches (N)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))                                # [1, 1, D]
        # 수식 (1) 의 [x_class]

        self.patch_embedding_projection = nn.Conv2d(3, self.dim, self.patch_size, stride=self.patch_size)  # projection
        # 수식 (1) 의 [x_p^1E; x_p^2E; ..., ;x_p^N]

        self.position_embedding = nn.Parameter(torch.empty(1, (self.num_patches + 1), self.dim))  # [1, N + 1, D]

        self.position_embedding_1 = nn.Parameter(torch.empty(1, 1, self.dim))  # [1, N + 1, D]
        self.position_embedding_ = nn.Parameter(torch.empty([1, 1, image_size // patch_size, image_size // patch_size]))
        self.pos_embed_conv = nn.Conv2d(1, self.dim, kernel_size=3, stride=3, padding=5)
        # 수식 (1) 의 E_pos
        torch.nn.init.normal_(self.position_embedding, std=.02)  # 확인해보기

        torch.nn.init.normal_(self.position_embedding_1, std=.2)  # 확인해보기
        torch.nn.init.normal_(self.position_embedding_, std=.2)  # 확인해보기
        # in the paper, they refer to "we use standard learnable 1D positional embeddings"
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim=self.dim, layers=layers, heads=heads, mlp_size=mlp_size, dropout=dropout)
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, img):
        batch_size = img.size(0)
        x = self.patch_embedding_projection(img)                                     # [B, 64, 28, 28]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_patches, self.dim)  # [B, 784, 64]

        # ---------------------------- ** change the order of transformer ** ------------------------------
        # ** random order **
        # indices = torch.arange(self.num_patches)
        # torch.manual_seed(0)
        # perm = torch.randperm(len(indices))
        # # print(perm)
        # new_order_indices = indices[perm]
        # x = x[:, new_order_indices, :]

        num_one_side_elements = int(self.num_patches ** 0.5)

        # ** reverse order **
        # new_order_indices = torch.flip(torch.arange(self.num_patches).unsqueeze(0), dims=(0, 1)).squeeze()
        # # print(new_order_indices) tensor([15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
        # x = x[:, new_order_indices, :]

        # **  vertical order (from up to down) **
        # new_order_indices = torch.arange(self.num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        # # print(new_order_indices) tensor([ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15])
        # x = x[:, new_order_indices, :]

        # **  vertical order (from down to up) **
        # new_order_indices = torch.arange(self.num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        # new_order_indices = torch.flip(new_order_indices.unsqueeze(0), dims=(0, 1)).squeeze()
        # # print(new_order_indices) tensor([15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0])
        # x = x[:, new_order_indices, :]
        # ---------------------------- ** end of change the order of transformer ** ------------------------------

        # ---------- 수식 (1) -------------
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)                       # expand to [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)                                        # [B, 1 + N, D]
        # patch embeddings = [x_class; x_p^1E; x_p^2E; ..., ;x_p^N]

        # ** conv pos embedding **
        # pos_embed = self.pos_embed_conv(self.position_embedding_)
        # pos_embed = torch.cat([pos_embed.view(1, self.dim, self.num_patches).permute(0, 2, 1), self.position_embedding_1], dim=1)
        # x += pos_embed

        x += self.position_embedding
        # print(x.shape)
        # [x_class; x_p^1E; x_p^2E; ..., ;x_p^N] + E_pos

        # ---------- 수식 (2) -------------
        x = self.dropout(x)
        x = self.transformer(x)                                             # [B, 257, D]

        # -------------- 맨 앞에꺼 가져오는부분 --------------
        x = x[:, 0]
        # x = torch.mean(x, dim=1)
        x = self.nn1(x)
        return x


if __name__ == '__main__':
    image = torch.randn([2, 3, 32, 32])
    vit = VisionTransformer(image_size=32, dim=384, heads=12, layers=7, mlp_size=384, patch_size=4)
    output = vit(image)
    print(output.size())
