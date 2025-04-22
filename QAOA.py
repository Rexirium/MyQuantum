from qucircuit import*
from scipy.optimize import minimize
from classical import MinObjective
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

rng=np.random.default_rng()
# Make Ising Hamiltonian of H=\sum_{i,j}J_{ij} Z_i Z_j
def makeIsingHamiltonian(Lsize:int, weights=None):
    hamiltonian = []
    for i in range(0, Lsize, 2):
        operate = Operator('ZZ', (i+1, i+2))
        if weights is not None:
            operate.coeff = weights[i]
        hamiltonian.append(operate)      
    for i in range(1, Lsize, 2):
        operate = Operator('ZZ', (i+1, (i+2)%Lsize))
        if weights is not None:
            operate.coeff = weights[i]
        hamiltonian.append(operate)
    return hamiltonian

# Make transverse field Ising Hamiltonian H=\sum_{i,j}J_{ij}Z_i Z_j-\sum_i X_i
def makeTFIsingHamiltonian(Lsize:int, weightZ=None, weightX=None):
    hamiltonian = makeIsingHamiltonian(Lsize, weightZ)
    TFoperators = Operator('TField', 0) # the TFoperators.position = 0 have no use.
    if weightX is not None:
        TFoperators.coeff = weightX   # weightX should be an np.ndarray, so is weightZ, if not None
    hamiltonian.append(TFoperators)
    return hamiltonian

# Generate the target hamiltonian from a weighted graph.
def makeHamiltonian(graph:nx.Graph):
    hamilton = []
    for (u,v,wt) in graph.edges.data():
        if 'name' in wt:
            op = Operator(graph['name'], (u+1, v+1))
        else:
            op = Operator('ZZ', (u+1, v+1))
        op.coeff = wt['weight']
        hamilton.append(op)
    return hamilton

# Cost function evaluated as F(theta)=<psi(theta)|H_c|psi(theta)>
def CostFunction(theta:np.ndarray, depth:int, qustate:QuRegister, hamil_cost:list[Operator]):
    qustate.initalize(inistate = '+') # initialized as \otimes_i |+_i>=1/sqrt{dim}*\sum_z |z>
    for layer in range(depth):  # theta=(gamma,beta)
        qustate.Evolving(hamil_cost, theta[layer])
        qustate.TField_evolve(theta[layer + depth])
    return Expectation(qustate, hamil_cost)

# QAOA procedure of one round
def QAOA_procedure(hamil_cost, Lsize:int, depth:int, method = 'BFGS'):
    qustate = QuRegister(Lsize)
    theta0 = rng.uniform(0., ma.tau, size = 2*depth)
    runtime = 0.
    def Runtime_sum(xk):
        nonlocal runtime
        runtime += sum(xk)
    Min = minimize(CostFunction, x0 = theta0, args=(depth, qustate, hamil_cost), method=method,\
        callback = Runtime_sum)
    return Min.fun, Min.x, runtime

if __name__=='__main__':
    L, p = 10, 5
    graph = nx.Graph()
    for (u,v) in nx.random_regular_graph(3, L).edges():
        graph.add_edge(u, v, weight = 1)
    nx.draw_networkx(graph)
    plt.show()
    Cmin = MinObjective(L, graph)
    Hamil = makeHamiltonian(graph)
    HamilIsing = makeIsingHamiltonian(L)
    start = time.perf_counter()
    Res = QAOA_procedure(Hamil, L, p, method = 'BFGS')
    alpha = abs(Res[0]/Cmin[0])
    end = time.perf_counter()
    print(Res)
    print(Cmin)
    print(alpha)
    print("Running time: {:5f} s".format(end-start))
        