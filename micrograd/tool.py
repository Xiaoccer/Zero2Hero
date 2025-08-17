from graphviz import Digraph

def trace(root):
    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ('LR', 'TB')
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        dot.node(name=str(id(n)),label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape="record")
        if n.op:
            dot.node(name=str(id(n)) + n.op , label = n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot