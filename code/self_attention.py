import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 三个线性层：分别把输入映射成 Q 查询、K 键、V 值
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        # x shape: [B, L, D]
        # B=batch批次大小，L=序列长度，D=每个token向量维度
        B, L, D = x.shape

        # 投影得到 Q K V
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # ========== 缩放点积注意力核心公式 ==========
        # Q @ K^T ：Q 和 K 的转置做点积，得到相似度分数矩阵 [B, L, L]
        # / sqrt(D) 缩放：维度D越大，点积值方差越大，softmax会变尖锐，缩放稳住梯度
        score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(D, dtype=torch.float32))

        # 掩码 mask：用来遮住未来位置（解码器自注意力，不能看到后面的token）
        if mask is not None:
            # mask=True 的位置填充 -1e9，softmax之后几乎等于0
            score = score.masked_fill(mask, -1e9)

        # softmax 归一化，得到注意力权重（每行总和=1）
        attn = F.softmax(score, dim=-1)

        # 权重 × V向量：加权求和，融合所有token信息
        out = torch.matmul(attn, V)
        return out, attn

# 测试代码
if __name__ == "__main__":
    # 2个样本，每个序列5个token，每个token维度16
    x = torch.randn(2, 5, 16)
    # 上三角掩码：diagonal=1，对角线右上全部置1，遮住未来token
    mask = torch.triu(torch.ones(5,5), diagonal=1).bool()
    attn_layer = SingleHeadAttention(16)
    res, w = attn_layer(x, mask)
    print(res.shape)
