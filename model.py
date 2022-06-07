import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)  # product of v times whatever inside softmax
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)
        self.head_dim = int(dim // num_heads)
        self.wQs = []
        self.wKs = []
        self.wVs = []
        self.wO = nn.Linear(self.dim, self.dim)
        self.attention = Attention(self.dim)
        for i in range(self.num_heads):
            self.wQs.append(nn.Linear(self.head_dim, self.head_dim))
            self.wKs.append(nn.Linear(self.head_dim, self.head_dim))
            self.wVs.append(nn.Linear(self.head_dim, self.head_dim))
        torch.nn.init.xavier_uniform_(self.wO.weight)
        torch.nn.init.zeros_(self.wO.bias)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        xs = torch.split(x, self.head_dim, dim=-1)  # [B, 16, 384] -> tuple{(V, 16, 32)} * 12
        head = []
        for i in range(self.num_heads):
            q = self.wQs[i](xs[i])  #
            k = self.wKs[i](xs[i])
            v = self.wVs[i](xs[i])
            head.append(self.attention(q, k, v))    # [2, 16, 32]
        head = torch.cat(head, dim=-1)              # [2, 16, 384]
        x = self.wO(head)                           # [2, 16, 384]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim:int, mlp_dim:int, num_heads:int=12, dropout_ratio:float=0.1):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(dim)
        self.msa = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.la2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class ViT(nn.Module):
    def __init__(self, patch_size, image_size, num_layers, dim, mlp_dim, num_heads, dropout_ratio, num_classes,
                 is_cls_token):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size                                                              # 1 // 4
        self.num_patches = (image_size // patch_size) ** 2                                        # 784

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3                 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, dim)  # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, dim))
        enc_list = [TransformerEncoder(dim, mlp_dim=mlp_dim, dropout_ratio=dropout_ratio, num_heads=num_heads) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes) # for cls_token
        )

    def forward(self, x):

        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    transformer = TransformerEncoder(dim=384, mlp_dim=384, num_heads=12, dropout_ratio=0.1)
    # msa = MultiHeadSelfAttention(dim=384, num_heads=12)
    x = torch.randn([2, 16, 384])
    x = transformer(x)
    print(x.size())