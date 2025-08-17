
class Value:
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.prev = set(children)
        self.op = op
        self.label = label

        self.grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward():
            self.grad += out.grad # *  1
            other.grad += out.grad # * 1
        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad* self.data
        out._backward = backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def backward():
            self.grad += out.grad * (other * self.data**(other-1))
        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other): # other - self
        return -self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other):
        return self * other**-1 # 经典

    def __rtruediv__(self, other): # other / self
        return (self**-1) * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'Relu')

        def backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = backward
        return out

    # 第一次实现忘了要拓扑排序，主要是性能递归会有问题
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 记得将 gard 设为 1
        self.grad = 1
        for v in reversed(topo):
            v._backward()

if __name__ == "__main__":
    from tool import draw_dot
    a = Value(1.0)
    b = 3 * a
    c = a + b
    e = 10 - c
    f = e **2
    g = f.relu()
    g.backward()
    dot = draw_dot(g)
    dot.render('output', view=True)
