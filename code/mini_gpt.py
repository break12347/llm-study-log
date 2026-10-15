import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
class MultiHeadAttention(nn.Module):
    def __init__(self, D,n_heads):
        """
        D:模型隐藏层维度(embedding维度)
        n_heads:注意力头数量,D必须可以被n_heads整除
        """
        super().__init__()
        assert D%n_heads==0
        self.D=D
        self.n_heads=n_heads
        self.d_head=D // n_heads# 每个头的维度
        # 一次性把x投影成 q,k,v 三个向量，合并成 3*D
        self.qkv_proj = nn.Linear(D, 3 * D)
        # 多头计算完成后，合并输出的线性层
        self.out_proj = nn.Linear(D, D)
    def forward(self,x,mask=None):
         # x: [B, L, D]
        # B=batch批次大小，L=序列长度，D=隐藏维度
        B, L, D = x.shape
        # 步骤1：一次性得到 q k v
        qkv = self.qkv_proj(x) # [B, L, 3D]
        q, k, v = torch.split(qkv, self.D, dim=-1) # 在最后一维切分成三份，每份D
        # 步骤2：分头，从 [B,L,D] → [B, n_head, L, d_head]
        # transpose交换L和头的维度，方便后面矩阵相乘
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1,2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1,2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1,2)
        # 步骤3：注意力分数 Q @ K^T / sqrt(d_k)
        attn_score = q @ k.transpose(-1,-2) / (self.d_head ** 0.5)
        # mask：GPT的因果掩码，掩盖未来token，不能看到后面的字
        if mask is not None:
            # 把mask=0的位置填充负无穷，softmax之后概率趋近0
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1) # 在最后一维做softmax得到权重
        # 步骤4：权重乘value
        out = attn_weight @ v
        # 步骤5：合并多头，还原维度
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(out)


class LayerNorm(nn.Module):
    def __init__(self,D,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(D)) # γ
        self.shift=nn.Parameter(torch.zeros(D)) # β
    def forward(self,x):
        #x:[B,L,D],对最后一维（特征维度D）归一化
        mean=x.mean(dim=-1,keepdim=True) # keepdim保持维度，方便广播
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        x_norm=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*x_norm+self.shift
class FeedForward(nn.Module):
    def __init__(self,D):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(D,4*D),
            nn.GELU(),
            nn.Linear(4*D,D)
        )
    def forward(self,x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self,D,n_heads):
        super().__init__()
        self.ln1=LayerNorm(D)
        self.attn=MultiHeadAttention(D,n_heads)
        self.ln2=LayerNorm(D)
        self.ffn=FeedForward(D)
    def forward(self,x,mask):
        x=x+self.attn(self.ln1(x),mask)
        x=x+self.ffn(self.ln2(x))
        return x
class MiniGPT(nn.Module):
    def __init__(self, vocab_size,D,n_heads,n_blocks,max_len):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,D)
        self.pos_emb=nn.Embedding(max_len,D)
        self.blocks=nn.Sequential(*[TransformerBlock(D,n_heads) for _ in range(n_blocks)])
        self.In_final=LayerNorm(D)
        self.lm_head=nn.Linear(D,vocab_size)# 语言头：D映射回词表维度，预测下一个token
        self.max_len=max_len#模型支持最大序列长度
    def forward(self,x):
        # x：输入token id，shape [B, L]
        B,L=x.shape
        pos=torch.arange(0,L,device=x.device).unsqueeze(0)
        tok_emb=self.token_emb(x)
        pos_emb=self.pos_emb(pos)
        x=tok_emb+pos_emb
        mask=torch.tril(torch.ones(L,L,device=x.device))
        for blk in self.blocks:
            x=blk(x,mask)
        x=self.In_final(x)
        logits=self.lm_head(x)# [B, L, vocab_size]，每个位置对词表打分
        return logits
    @torch.no_grad() # 推理阶段不需要计算梯度，节省显存
    def generate(self,idx,max_new_tokens,temperature=1.2,top_k=20, rep_penalty=1.1):
        """
        idx: [B, L] 初始上下文token序列
        max_new_tokens:需要新增生成多少个token
        """
        for _ in range(max_new_tokens):
            # 如果序列超过max_len，只保留最后max_len个token
            idx_cond=idx[:,-self.max_len:]
            logits=self(idx_cond)
            # 只取最后一个位置的logits，用来预测下一个词
            logits_last=logits[:,-1,:]
            temperature = 1.2
            logits_last = logits_last / temperature
                # top-k 截断
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)), dim=-1)
                cutoff = v[:, -1].unsqueeze(-1)
                logits_last = logits_last.masked_fill(logits_last < cutoff, -1e9)

        # 重复惩罚
            for token_id in idx[0].tolist():
                logits_last[0, token_id] /= rep_penalty
            probs=F.softmax(logits_last,dim=-1)
            next_token=torch.argmax(probs,dim=-1,keepdim=True)
            idx=torch.cat([idx,next_token],dim=1)
        return idx
if __name__ == "__main__":
    # 超参数定义
    vocab_size = 1000   # 词表大小
    D = 128             # 隐藏维度
    n_heads = 8         # 注意力头
    n_blocks = 4        # transformer块层数
    max_len = 64        # 最大序列长度
    batch_size = 8

    # 自动选择GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, D, n_heads, n_blocks, max_len).to(device)
    opt = AdamW(model.parameters(), lr=3e-4) # AdamW优化器，GPT标配
    loss_fn = nn.CrossEntropyLoss() # 交叉熵损失

    # 模拟训练数据：输入x，任务是预测移位后的下一个token（语言建模CLM）
    x = torch.randint(0, vocab_size, (batch_size, max_len), device=device)
    for epoch in range(100):
        logits = model(x) # 前向传播 [B,L,V]
        # CrossEntropyLoss输入要求：(N, C) 与 (N)，需要展平
        logits_flat = logits.view(-1, vocab_size)
        y_flat = x.view(-1) # CLM任务：用当前token预测下一个，这里简化演示
        loss = loss_fn(logits_flat, y_flat)

        opt.zero_grad() # 清空梯度
        loss.backward() # 反向传播求梯度
        opt.step()      # 更新参数

        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss: {loss.item():.4f}")

    # 测试自回归生成
    start_tokens = torch.randint(0, vocab_size, (1,10), device=device)
    out = model.generate(start_tokens, max_new_tokens=20,temperature=1.2)
    print("生成token序列:", out[0])
    

# epoch 0, loss: 7.0829
# epoch 10, loss: 6.1000
# epoch 20, loss: 5.1933
# epoch 30, loss: 4.3121
# epoch 40, loss: 3.4748
# epoch 50, loss: 2.7007
# epoch 60, loss: 2.0092
# epoch 70, loss: 1.4211
# epoch 80, loss: 0.9594
# epoch 90, loss: 0.6340
# 生成token序列: tensor([408, 441, 360, 839, 353, 801,  31, 976, 172, 541, 541, 541, 326, 594,
#         594, 594, 594, 594, 594, 594, 594, 594, 594, 594, 594, 594, 594, 594,
#         594, 594], device='cuda:0')



# epoch 0, loss: 7.0865
# epoch 10, loss: 6.1186
# epoch 20, loss: 5.2097
# epoch 30, loss: 4.3338
# epoch 40, loss: 3.4985
# epoch 50, loss: 2.7162
# epoch 60, loss: 2.0119
# epoch 70, loss: 1.4127
# epoch 80, loss: 0.9447
# epoch 90, loss: 0.6195
# 生成token序列: tensor([219, 282, 990, 733, 304,  62, 465, 177, 377, 425, 432, 432, 432, 432,
#         432, 554, 554, 978, 978, 978, 346, 346, 346, 694, 694, 694, 694, 957,
#         957, 957], device='cuda:0')