import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()
print(words[:8])

# 构建映射
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i ,s in enumerate(chars)}
# 技巧，一个字符标志开始或结束
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# 构建数据集
block_size = 3 # 使用 3 个字符来预测下一个字符

def build_dataset(words):
    X = []
    Y = []
    for w in words:
        # 初始化字符为 `...`
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

'''
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .
... ---> o
..o ---> l
.ol ---> i
oli ---> v
liv ---> i
ivi ---> a
via ---> .
'''
# build_dataset(words[:2])

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

# 训练集、验证集和测试集的划分
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
# 第一层：每个输入字符映射 10 维的向量。可以理解为一个 map
C = torch.randn((27, 10), generator=g)
# 第二层，输入 3 个字符，每个映射向量拼接一起，30 维
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
# 输出层
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))

# 记得设置梯度
for p in parameters:
  p.requires_grad = True

# 设置不同迭代次数的学习率
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(200000):

    # 随机挑选 32 个样本当做 batch
    ix = torch.randint(0, Xtr.shape[0], (32, ))

    # 技巧，tenser 能直接索引
    emb = C[Xtr[ix]] # (32, 3, 10)
    # concat 技巧 emb.view(-1, 30)，先拼成 (32, 30)
    # 隐含层需要 `tanh`
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
    logits = h @ W2 + b2 # (32, 27)
    # 用 `F.cross_entropy` 比手写公式的好处是内部做了防溢出的处理（每层的值会减去其 max 值）
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    # lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    stepi.append(i)
    # lossi.append(loss.item())
    # 使用对数输出使得输出比较平滑。比如即使 loss 值很大，对数后的值也比较小。，
    lossi.append(loss.log10().item())
    if i % 1000 == 0:
        print(loss.item())

plt.plot(stepi, lossi)
# plt.show()

# 输出训练集的 loss
emb = C[Xtr] # (xx, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(loss)

# 输出验证集的 loss
emb = C[Xdev] # (xx, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(loss)

# 输出测试集的 loss
emb = C[Xtest] # (xx, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytest)
print(loss)


# 用模型采样输出名字
for _ in range(20):

    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)] # (1, 3, 10)
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
        logits = h @ W2 + b2 # (1, 27)
        # 输出用 `softmax`
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))


# 若每个字符对应两维的向量，可以用散点图来可视化每个字符对应的特征
def visual():
    g = torch.Generator().manual_seed(2147483640)
    C = torch.randn((27, 2), generator=g)
    W1 = torch.randn((6, 200), generator=g)
    b1 = torch.randn(200, generator=g)
    W2 = torch.randn((200, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    for i in range(2000):
        ix = torch.randint(0, Xtr.shape[0], (32,))
        emb = C[Xtr[ix]]  # (32, 3, 2)
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # (32, 200)
        logits = h @ W2 + b2  # (32, 27)
        loss = F.cross_entropy(logits, Ytr[ix])
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 1000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
    print(loss)
    plt.figure(figsize=(8, 8))
    # tensor.data 返回新的一个 tensor
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        # tensor.item() 返回一个值
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
    # 显示网格
    plt.grid('minor')
    plt.show()

visual()