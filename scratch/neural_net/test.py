import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix
from scratch.neural_net import GCN
import torch

zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
labels = np.array([ 0 if zkc.nodes[node]['club'] == 'Mr. Hi' else 1 for node in zkc.nodes()])
y = torch.tensor(labels[:,None], dtype=torch.float64)

colors = []
for i in zkc.nodes():
    if labels[i] == 0:
        colors.append('blue')
    else:
        colors.append('red')

A = to_numpy_matrix(zkc, nodelist=order)

net = GCN(A)
I = torch.tensor(np.eye(net.N))


learning_rate = 1e-5

n_iter = 50000
losses = np.zeros(n_iter, dtype=float)
for i,t in enumerate(range(n_iter)):
    y_pred = net.forward(I)

    loss = ((y_pred - y)**2).sum()
    loss.backward()

    with torch.no_grad():
        net.W1 -= learning_rate*net.W1.grad
        net.W2 -= learning_rate*net.W2.grad

        net.W1.grad.zero_()
        net.W2.grad.zero_()

    losses[i] = loss.item()
