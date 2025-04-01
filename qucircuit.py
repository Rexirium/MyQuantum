import numpy as np
import math as ma
import secrets as scr
import time
from utilities import *

#entropy used for random sampling#
entropy = eval(f"0x{scr.randbits(128):x}")
rng = np.random.default_rng(entropy)

#Weak measurement initial state of probe with correlation strength lam and width sigma#
def phi_distr(x:float, sigma:float)->float: #square root of gaussian distribution centered at x=0
    factor = 1./ma.sqrt(ma.sqrt(ma.tau)*sigma)  
    exponent = -x*x/(4.*sigma*sigma)
    return factor * ma.exp(exponent)

def gaussian_distr(x, mu:float, sigma:float): # gaussian distribution
    factor = 1./(ma.sqrt(ma.tau)*sigma)
    exponent = -1./(2*sigma*sigma)*(x-mu)*(x-mu)
    return factor*np.exp(exponent)

# single qubit operator or double qubit operator 
class Operator:
    def __init__(self, name:str, pos):
        self.name = name  #name of operator 'Z','X','ZZ','CX','TField'
        self.position = pos # position to act on
        self.coeff = 1. # coefficient multiplied before the operator. float if operator is Hermitian
    def getName(self):
        return self.name
    def getPosition(self):
        return self.position
    def getCoeff(self):
        return self.coeff   

class QuRegister:
    #represent quantum state with an complex array of dimension 2^L, c[z] is the coefficent of basis 
    # |z> = |binary_repr(z)>
    def __init__(self,Lsize:int=0):
        self.num_qubits = Lsize
        self.dim = 1<<Lsize
        self.basis = np.arange(self.dim)
        self.state = np.zeros(self.dim, dtype=complex)
    def initalize(self, inistate='right'):
        if isinstance(inistate, str):
            if inistate == 'right':   # |psi0> = \otimes_i |+_i> = 1/sqrt{dim}\sum_z^(2^Lsize) |z>
                integer = 1<<(self.num_qubits//2)
                sqrt2 = 1. if self.num_qubits%2==0 else ma.sqrt(2)
                normalize = 1./(integer*sqrt2)
                self.state = normalize*np.ones(self.dim, dtype=complex)
            elif inistate == 'up': #|psi0> = |0>=|00...00>
                self.state = np.zeros(self.dim,dtype=complex)
                self.state[0] = complex(1.0, 0.0)
            elif inistate == 'GHZ': #1/sqrt(2)*(|00...0> +|11...1>)
                sqrt2 = 1./ma.sqrt(2)
                self.state = np.zeros(self.dim,dtype=complex)
                self.state[0], self.state[-1] = complex(sqrt2, 0.0), complex(sqrt2, 0.0)
            else:
                print("Wait for later development")
        elif isinstance(inistate, np.ndarray):
            normalize = 1./ma.sqrt(Norm2(inistate))
            self.state = normalize*inistate.astype(complex)
            self.dim = len(inistate)
            self.basis = np.arange(self.dim)
            self.num_qubits = len(bin(self.dim))-3
        else:
            print("Wait for later development")
        return
    
    def Normalize(self):
        normalize = 1/ma.sqrt(Norm2(self.state))
        self.state *= normalize
    # get the basis of ind, in the form of a binary string
    def getbasis(self, ind:int)->str:
        return np.binary_repr(ind, width=self.num_qubits)
    # the coefficient of a basis 
    def getcoeff(self, basis:int|str)->complex:
        if isinstance(basis, int):
            ind = basis
        else:
            ind = int(basis,2)
        return self.state[ind]
    def totalState(self): # return a copy of state
        return self.state.copy()
    # direct product one more qubit, default on the right.
    def getNorm(self):
        return ma.sqrt(Norm2(self.state))
    
    def addCartesian_prod(self, substate:np.ndarray, right=True):
        dim_add = len(substate)
        self.num_qubits += len(bin(dim_add))-3
        self.dim *= dim_add
        self.basis = np.arange(self.dim)
        if right:
            self.state = np.kron(self.state, substate)
        else:
            self.state = np.kron(substate, self.state)
    
    # gates apply at position 'pos' acting on a state 'state' in dimension 'dim' space
    # pos means the pos-th qubit in the system, pos = 1,2,3,...,num_qubits
    # correspond to the dig means dig-th digit in the binary_repr of basis
    # pos + dig = num_qubits
    # retstate to control whether to only return the new state (True) 
    # or to directly modify the state (False)
    
    def Zgate(self, pos:int, retstate=False):
        if retstate == True:
            return self.state*onebit_sign(self.num_qubits - pos, self.basis)
        else:
            self.state *= onebit_sign(self.num_qubits - pos, self.basis)
            return
    # exchange the coefficient of base |zold> and |znew>
    def Xgate(self, pos:int, retstate=False):
        zold, znew = flipp_pair(self.num_qubits - pos, self.basis)
        if retstate:
            newstate = np.zeros(self.dim, dtype=complex)
            newstate[zold], newstate[znew] = self.state[znew], self.state[zold]
        else:
            self.state[zold], self.state[znew] = self.state[znew], self.state[zold]
            return
    def Ygate(self, pos:int, retstate=False):
        zold,znew = flipp_pair(self.num_qubits-pos, self.basis)
        if retstate:
            newstate = np.zeros(self.dim, dtype=complex)
            newstate[zold], newstate[znew] = -1j*self.state[znew], 1j*self.state[zold]
            return newstate
        else:
            self.state[zold], self.state[znew] = -1j*self.state[znew], 1j*self.state[zold]
            return
    def Phgate(self, pos:int, phase:float, retstate=False): #Phase gate
        bits = onebit(self.basis, self.num_qubits - pos)
        mask = (bits==1)
        ephase = np.exp(1j*phase)
        if retstate:
            newstate = self.state.copy()
            newstate[mask] *= ephase
            return newstate
        else:
            self.state[mask] *= ephase
            return
    # Hadmard gate at 'pos'
    def Hgate(self, pos:int, retstate=False):
        co = 1./ma.sqrt(2)
        zold, znew = flipp_pair(self.num_qubits - pos, self.basis)
        statplus = self.state[zold] + self.state[znew]
        statminus = self.state[zold] - self.state[znew]
        if retstate:
            newstate=np.zeros(self.dim,dtype=complex)
            newstate[zold], newstate[znew] = co*statplus, co*statminus
            return newstate
        else:
            self.state[zold], self.state[znew] = co*statplus, co*statminus
            return
    # two bit gate of ZZ XX
    def ZZgate(self, pos:tuple, retstate=False):
        dig = self.num_qubits - pos[0], self.num_qubits - pos[1]
        if retstate:
            return self.state*twobits_sign(dig, self.basis)
        else:
            self.state *= twobits_sign(dig, self.basis)
            return
    def XXgate(self, pos:tuple, retstate=False):
        dig = self.num_qubits-pos[0], self.num_qubits-pos[1]
        zold, znew = flipp_twopairs(dig, self.basis)
        if retstate:
            newstate = np.zeros(self.dim, dtype=complex)
            newstate[zold], newstate[znew] = self.state[znew], self.state[zold]
        else:
            self.state[zold], self.state[znew] = self.state[znew], self.state[zold]
            return
    # Controlled gate of X and Z, with control qubit and target qubit pos=(cpos,tpos)
    def CXgate(self, pos:tuple, retstate=False):
        dig = self.num_qubits - pos[0], self.num_qubits - pos[1]
        zold,znew = flipp_pair(dig[1], self.basis, control=dig[0])
        if retstate:
            newstate = self.state.copy()
            newstate[zold], newstate[znew] = self.state[znew], self.state[zold]
            return newstate
        else:              
            self.state[zold], self.state[znew] = self.state[znew], self.state[zold]
            return
    def CYgate(self, pos:tuple, retstate=False):
        dig = self.num_qubits - pos[0], self.num_qubits - pos[1]
        zold, znew = flipp_pair(dig[1], self.basis, control = dig[0])
        if retstate:
            newstate = self.state.copy()
            newstate[zold], newstate[znew] = -1j*self.state[znew], 1j*self.state[zold]
            return newstate
        else:
            self.state[zold], self.state[znew] = -1j*self.state[znew], 1j*self.state[zold]
            return
    def CZgate(self, pos:tuple, retstate=False):
        control = onebit(self.basis, self.num_qubits-pos[0])
        target = onebit(self.basis, self.num_qubits-pos[1])
        bits = control&target
        signs = np.where(bits==1, -1., 1.)
        if retstate:
            return self.state*signs
        else:
            self.state *= signs
            return 
    # acting an arbitrary hamiltonian on the state. Hamiltonian as a list of Operator type
    def Acting(self, hamilton:list[Operator], retstate=False):
        finalstate=np.zeros(self.dim, dtype=complex)
        for op in hamilton:
            name = op.name
            if name == 'ZZ':
                finalstate += op.coeff*self.ZZgate(op.position, retstate=True)
            elif name=='TField':
                finalstate += op.coeff*self.TFieldgate(0, retstate=True)
            else:
                finalstate += op.coeff*getattr(self,op.name + 'gate')(pos = op.position, retstate=True)
        if retstate:
            return finalstate
        else:
            self.state = finalstate

    # the acting of transverse field H=\sum_i X_i is slow using the method above due to copy frequently 
    # , TField gate is specialized so can do it faster.TField means H=-\sum_i X_i
    def TFieldgate(self, pos=-1, retstate=False): 
        finalstate = np.zeros(self.dim, dtype=complex)
        newstate = np.zeros(self.dim, dtype=complex)
        for dig in range(self.num_qubits):
            zold, znew = flipp_pair(dig, self.basis)
            newstate[zold], newstate[znew] = self.state[znew], self.state[zold]
            finalstate -= newstate
        if retstate:
            return finalstate
        else:
            self.state = finalstate
    # parametrized evolution of the form exp(-i para * gate)
    def Zevolve(self, pos:int, para:float):
        sign = onebit_sign(self.num_qubits-pos, self.basis)
        co, isi = ma.cos(para), ma.sin(para)*1j
        phase = co - isi*sign
        self.state *= phase
        return
    def Xevolve(self, pos:int, para:float):
        co, isi = ma.cos(para), 1j*ma.sin(para)
        zold,znew = flipp_pair(self.num_qubits-pos, self.basis)
        statold, statnew = self.state[zold], self.state[znew]
        self.state[zold], self.state[znew] = co*statold - isi*statnew,\
            isi*statold - co*statnew
        return
    def Yevolve(self, pos:int, para:float):
        co, si = ma.cos(para), ma.sin(para)
        zold, znew = flipp_pair(self.num_qubits-pos, self.basis)
        statold, statnew = self.state[zold], self.state[znew]
        self.state[zold], self.state[znew] = co*statold + si*statnew,\
            -si*statold + co*statnew
        return
    def ZZevolve(self, pos:tuple, para:float):
        co, isi = ma.cos(para), 1j*ma.sin(para)
        dig = self.num_qubits-pos[0], self.num_qubits-pos[1]
        sign = twobits_sign(dig, self.basis)
        phase = co - isi*sign # exp(-1j*para)
        self.state *= phase
        return
    def XXevolve(self, pos:tuple, para:float):
        co, isi = ma.cos(para), 1j*ma.sin(para)
        dig = self.num_qubits - pos[0], self.num_qubits-pos[1]
        zold, znew = flipp_twopairs(dig, self.basis)
        statold, statnew = self.state[zold], self.state[znew]
        self.state[zold], self.state[znew] = co*statold - isi*statnew,\
            isi*statold - co*statnew
        return
    def evolve(self, op:Operator, param:float):
        if (name := op.name) == 'TField':   # the TField operators should evolve together with 
            return                      # mixer hamiltonian
        para = param*op.coeff
        if name == 'ZZ':
            self.ZZevolve(op.position, para)
        else:
            getattr(self, name+'evolve')(pos = op.position,para = para) 
        return
    def Evolving(self, hamilton:list[Operator], param:float):
        for op in hamilton:
            if (name:=op.name) == 'TField':   # the TField operators should evolve together with 
                continue                 # mixer hamiltonian
            para = param*op.coeff
            if name == 'ZZ':
                self.ZZevolve(op.position, para)
            else:
                getattr(self,name + 'evolve')(pos = op.position, para = para)  
        return
    # TField means H_B=-\sum_i X_i , U(para)=exp(-iH_B)=exp(\sum_i X_i)   
    def TField_evolve(self, para:float):
        co, isi = ma.cos(para), 1j*ma.sin(para)
        for dig in range(self.num_qubits):
            zold,znew = flipp_pair(dig, self.basis)
            statold, statnew = self.state[zold], self.state[znew]
            self.state[zold], self.state[znew] = co*statold + isi*statnew,\
                isi*statold + co*statnew
        return
    # single qubit projection measurement of X Z operator
    def proj_measureZ(self, pos:int, evolve=True, kill=False, counts=1): # if evolve take True, then counts must be 1
        mask = onebit(self.basis, self.num_qubits-pos).astype(bool)
        filtered = self.state[mask]
        prob1 = Norm2(filtered) 
        prob0 = 1 - prob1
        res = rng.choice((0,1), p=(prob0,prob1))  # 0 :up, 1: down
        if evolve and kill == False:
            if res == 1:
                proj = (1./ma.sqrt(prob1))*mask
            else:
                proj = (1./ma.sqrt(prob0))*(~mask)
            self.state *= proj
        elif evolve and kill == True:  # kill the qubit measured, i.e. divide dimension of Hilbert space by 2
            if res == 1:
                self.state = (1./ma.sqrt(prob1))*filtered
            else:
                unfiltered = self.state[~mask]
                self.state = (1./ma.sqrt(prob0))*unfiltered
            self.dim >>= 1
            self.num_qubits -= 1
            self.basis = np.arange(self.dim)
        else: # no evolve and no kill
            res = rng.choice((0, 1), p=(prob0, prob1), size=counts)
        return res
    def proj_measureX(self, pos:int, evolve=True, kill=False, counts=1):
        zold, znew = flipp_pair(self.num_qubits-pos, self.basis)
        filtered = self.state[zold] + self.state[znew]
        probr = 0.5*Norm2(filtered)
        probl = 1. - probr
        res = rng.choice((0, 1), p = (probr, probl))  # 0: right, 1: left
        if counts > 1:
            evolve = False
        if evolve :
            if res == 0:
                spr = 0.5/ma.sqrt(probr)
                newcoeff = spr*(self.state[zold] + self.state[znew])
            else:
                spl = 0.5/ma.sqrt(probl)
                newcoeff = spl*(self.state[zold] - self.state[znew])
            if kill == False:
                self.state[zold], self.state[znew] = newcoeff, newcoeff
            else:
                self.state = (ma.sqrt(2)) * newcoeff
                self.num_qubits -= 1
                self.dim >>= 1
        else:
            res = rng.choice((0, 1), p=(probr, probl), size=counts)
        return res
    # weak measurement at position 'pos' on spin Z. return the coordinate of the probe
    def weak_measureZ(self, pos:int, lam:float, sigma:float, evolve=True, counts=1):
        mask = onebit(self.basis, self.num_qubits-pos).astype(bool)
        filtered = self.state[mask]
        prob1 = Norm2(filtered)
        prob0 = 1 - prob1
        if counts > 1:
            evolve = False
        if evolve:
            xi = rng.random()
            xres = rng.normal(lam, sigma) if xi < prob0 else rng.normal(-lam, sigma)
            phix0 = phi_distr(xres - lam, sigma)
            phix1 = phi_distr(xres + lam, sigma)
            probx = ma.sqrt(prob0*phix0*phix0 + prob1*phix1*phix1)
            phix0 /= probx
            phix1 /= probx
            collapse = np.where(mask, phix1, phix0)
            self.state *= collapse
        else:
            xis = rng.random(size = counts)
            lams = np.where(xis < prob0, lam, -lam)
            xres = rng.normal(lams, sigma, size = counts)
        return xres
    # weak measurement at position 'pos' on spin X. return the coordinate of the probe
    def weak_measureX(self, pos:int, lam:float, sigma:float, evolve=True, counts=1):
        zold, znew = flipp_pair(self.num_qubits-pos, self.basis)
        filtered=self.state[zold]+self.state[znew]
        probr = 0.5*Norm2(filtered)
        probl = 1. - probr
        if counts > 1:
            evolve = False
        if evolve:
            xi = rng.random()
            xres = rng.normal(lam, sigma) if xi < probr else rng.normal(-lam, sigma)
            phixr = phi_distr(xres - lam,sigma)
            phixl = phi_distr(xres + lam,sigma)
            probx = ma.sqrt(probr*phixr*phixr + probl*phixl*phixl)
            phixl /= probx
            phixr /= probx
            coeffr = 0.5*(self.state[zold] + self.state[znew])
            coeffl = 0.5*(self.state[zold] - self.state[znew])
            self.state[zold] = coeffr*phixr + coeffl*phixl
            self.state[znew] = coeffr*phixr - coeffl*phixl
        else:
            xis = rng.random(size = counts)
            lams = np.where(xis < probr, lam, -lam)
            xres = rng.normal(lams, sigma, size = counts)
        return xres
# direct product of two subsystem.
def Cartesian_prod(qustate1:QuRegister, qustate2:QuRegister):
    num_tot = qustate1.num_qubits + qustate2.num_qubits
    qustate = QuRegister(num_tot)
    qustate.state = np.kron(qustate1.state, qustate2.state)
    return qustate
# Evaluating the expectation value : <psi |H|psi>
def Expectation(qustate:QuRegister, hamilton:list[Operator]):
    value = complex(0.0, 0.0)
    TFflag = False
    for op in hamilton:
        if (name:=op.name) == 'ZZ':
            newstate = qustate.ZZgate(op.position, retstate=True)
        elif name == 'TField':
            TFflag = True
        else:
            newstate = getattr(qustate, name+'gate')(pos = op.position, retstate=True)
        value += op.coeff*np.dot(qustate.state.conj(), newstate)
    if TFflag:
        for dig in range(qustate.num_qubits):
            zold, znew = flipp_pair(dig,qustate.dim)
            value -= 2*np.dot(qustate.state[znew].conj(), qustate.state[zold])
    return value.real
            
       
if __name__=='__main__':
    import matplotlib.pyplot as plt
    p0, p1=1./4, 3./4
    alpha, beta = ma.sqrt(p0/2) + ma.sqrt(p1/2), ma.sqrt(p0/2) - ma.sqrt(p1/2)
    qr = QuRegister(1)
    qr.state = np.array((alpha,beta), dtype=complex)
    start = time.perf_counter()
    results = qr.weak_measureX(1, 2., 1., evolve=False, counts=10000)
    num_bins = 40
    fig, ax =plt.subplots()
    n, bins, patches = ax.hist(results, num_bins, density=True)
    probxs = p0*gaussian_distr(bins, 2., 1.) + p1*gaussian_distr(bins, -2., 1.)
    ax.plot(bins,probxs, '--')
    fig.tight_layout()
    plt.show()
    end = time.perf_counter()
    print("Running time is {:.6f} ms".format(1000*(end-start)))
    