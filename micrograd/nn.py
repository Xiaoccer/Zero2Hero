import random
from engine import Value

class Module:

    # 每轮梯度更新后，需重置
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

# 每层里的一个神经元
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # 需要梯度
        self.b = Value(0) # 需要梯度
        # self.b = Value(random.uniform(-1, 1))

        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# 一层有多个神经元
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# 多层，注意初始时是输入是一个数组，指定每层有多少个神经元
class MLP(Module):
    def __init__(self, nin, nouts, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin= i!= len(nouts) -1)  for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layers in self.layers for p in layers.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layers) for layers in self.layers)}]"


if __name__ == "__main__":
    x = [1 ,2 ,3]
    mlp = MLP(3, [1, 3, 2, 1])
    print(mlp)
    print(mlp(x))

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
    n = MLP(3, [4, 4, 1])
    for k in range(20):

        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass
        mlp.zero_grad() # 记得
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.01 * p.grad # 反向更新

        print(k, loss.data)
    print(ypred)