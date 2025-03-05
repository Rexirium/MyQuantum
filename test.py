import time
import numpy as np
import math as ma
import networkx as nx
import matplotlib.pyplot as plt
from qucircuit import *
rng=np.random.default_rng()

####
Alice=QuRegister(1)
alpha=complex(*rng.random(2))
beta=complex(*rng.random(2))
Alice.initalize(np.array([alpha,beta]))
Bob=QuRegister(2)
Bob.initalize(inistate='GHZ')
print(Alice.state)
start=time.perf_counter()
qc=Cartesian_prod(Alice,Bob)
qc.CXgate((1,2))
qc.Hgate(1)
res1=qc.proj_measureZ(1,kill=True)
res2=qc.proj_measureZ(1,kill=True)
if res2:
    qc.Xgate(1)
if res1:
    qc.Zgate(1)
end=time.perf_counter()
print(qc.state)
print("Running time is {:.6f} ms".format(1000*(end-start)))