import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utilities import *

def MinObjective(Lsize:int, graph:nx.Graph):
    dim = 1<<Lsize
    summ = np.zeros(dim)
    for (u,v,wt) in graph.edges.data('weight'):
        sign = twobits_sign((u,v), np.arange(dim))
        summ += wt*sign
    Cmin = np.min(summ)
    min_config = np.where(summ==Cmin)[0]
    return Cmin, min_config

if __name__=='__main__':
    graph = nx.Graph()
    for (u,v) in nx.random_regular_graph(3,10).edges():
        graph.add_edge(u, v, weight = 1)
    nx.draw_networkx(graph)
    plt.show()
    print(MinObjective(10, graph))
    
        
            
    