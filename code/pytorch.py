import torch
import torch.nn as nn

print("=== 0. 环境 ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print()
print("=== 1. Tensor 基础 ===")
a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print("a.shape =", a.shape)
print("a.dtype =", a.dtype)
print("a.sum() =", a.sum(), "  shape =", a.sum().shape)
print("a.mean(dim=0) =", a.mean(dim=0))
print("a.mean(dim=-1) =", a.mean(dim=-1))

print()
print("=== 2. nn.Linear：权重长什么样 ===")
lin = nn.Linear(4, 3)
print("lin.weight.shape =", lin.weight.shape)
print("lin.bias.shape   =", lin.bias.shape)
x = torch.randn(2, 5, 4)
print("输入 shape =", x.shape, "-> 输出 shape =", lin(x).shape)

print()
print("=== 3. nn.Embedding：查表 ===")
emb = nn.Embedding(10, 4)
ids = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])
print("ids.shape =", ids.shape)
print("emb(ids).shape =", emb(ids).shape)
print("emb 权重表 shape =", emb.weight.shape)

print()
print("=== 4. autograd：梯度是自动算的 ===")
w = torch.tensor([2.0], requires_grad=True)
y = (w * 3).sum()
y.backward()
print("y = w*3 =", y.item(), "  w.grad =", w.grad.item())

print()
print("=== 5. 训练循环五件套（用玩具任务：学会 y = 2x + 1）===")
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

xs = torch.linspace(-3, 3, 64).unsqueeze(1)
ys = 2 * xs + 1

for step in range(300):
    optimizer.zero_grad()
    pred = model(xs)
    loss = loss_fn(pred, ys)
    loss.backward()
    optimizer.step()
    if step % 60 == 0:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

print("x=0 时模型预测 =", round(model(torch.tensor([[0.0]])).item(), 3), "（正确答案 1.0）")

print()
print("=== 6. 交叉熵：下一个词预测用的损失 ===")
torch.manual_seed(1)
B, L, vocab = 2, 5, 10
logits = torch.randn(B, L, vocab)
target = torch.randint(0, vocab, (B, L))
loss = nn.CrossEntropyLoss()(logits.view(-1, vocab), target.view(-1))
print("logits.shape =", logits.shape, " target.shape =", target.shape)
print("随机猜测的交叉熵 =", round(loss.item(), 4), f"（ln(10) = {torch.log(torch.tensor(10.0)):.4f}）")
