import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

print(words[:10])

# 构建映射
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i ,s in enumerate(chars)}
# 技巧，一个字符标志开始或结束
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# 计数
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.'] # 技巧
    for ch1, ch2 in zip(chs, chs[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        N[i1, i2] += 1

# print(N)

# 技巧，更好的可视化
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, str(N[i, j].item()), ha='center', va='top', color='gray')
plt.axis('off')
# plt.show()

def generate_char_by_stats():
    g = torch.Generator().manual_seed(2147483647)

    P = (N+1).float() # +1 防止取 0 的对数为负无穷
    P /= P.sum(dim=1, keepdim=True) # 计算出下一个字符的概率

    for i in range(5):
        out = []
        ix = 0
        while True:
            p = P[ix]
            # 采样函数
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))


generate_char_by_stats()


def generate_char_by_model():
    # GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
    # equivalent to maximizing the log likelihood (because log is monotonic)
    # equivalent to minimizing the negative log likelihood
    # equivalent to minimizing the average negative log likelihood

    # log(a*b*c) = log(a) + log(b) + log(c)

    # 构建数据集
    xs = []
    ys = []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print('number of examples: ', num)

    # 构建模型
    g = torch.Generator().manual_seed(2147483647)
    # 随机化权重, 记得 `requires_grad=True`
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    # 训练
    for k in range(50):

        # 使用 one_hot 编码得到 [batches, 27], [batches, 27] * [27, 27] = [batches, 27]
        xenc= F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W # 输出的值范围(负无穷，正无穷)，因此可以人为定义该值含义为 log(统计值)。这种人为定义是精髓~
        counts = logits.exp() # 所以取指数就得到统计值
        probs = counts / counts.sum(dim=1, keepdim=True) # 得到了下一个字符的概念
        # btw: the last 2 lines here are together called a 'softmax'

        # 损失函数，取 batches 里每个标签值的概率，然后 minimizing the average negative log likelihood
        # `W` 参数 l2 正则化，防止过拟合
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
        print(loss.item())

        # 反向传播
        W.grad = None # 记得
        loss.backward()

        # 更新梯度
        W.data += -5.0 * W.grad


    # 预测
    for i in range(5):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))


generate_char_by_model()

