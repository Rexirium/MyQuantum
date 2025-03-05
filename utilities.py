import numpy as np
# read the bit of the digit 'dig' of the string num in binary representation
def onebit(num,dig:int):
    return (num&(1<<dig))>>dig
def twobits(num,dig:tuple):
    bits0=(num&(1<<dig[0]))>>dig[0]
    bits1=(num&(1<<dig[1]))>>dig[1]
    return (bits1<<1)|bits0
#read the sign of every basis at position 'pos' according to spin Z, 0:Z=1, 1:Z=-1
def onebit_sign(dig:int, dim:int):
    basis=np.arange(dim)
    bits=(basis&(1<<dig))>>dig
    return np.where(bits==0,1.,-1.)

# sign of every basis at two position correlated pos=(pos0,pos1)
def twobits_sign(dig:tuple,dim:int):
    basis=np.arange(dim)
    bits0=(basis&(1<<dig[0]))>>dig[0]
    bits1=(basis&(1<<dig[1]))>>dig[1]
    return np.where(bits0==bits1,1.,-1.)

## the most powerful improvement!! Pairing the basis according to a spin flip at position pos
# , breaking the indices 0,1,...,2^L-1 into two subarrays 
def flipp_pair(dig:int,dim:int,control=None):
    basis=np.arange(dim)
    mask1=((basis&(1<<dig))>>dig).astype(bool)
    mask0=~mask1
    if isinstance(control,int):
        cmask=((basis&(1<<control))>>control).astype(bool)
        mask0=mask0&cmask
        mask1=mask1&cmask
    return basis[mask0],basis[mask1]

def flipp_twopairs(dig:tuple,dim:int):
    basis=np.arange(dim)
    bits=twobits(basis,dig)
    mask0=(bits==0)
    mask1=(bits==1)
    mask2=(bits==2)
    mask3=(bits==3)
    lower=np.concatenate((basis[mask0],basis[mask1]))
    upper=np.concatenate((basis[mask3],basis[mask2]))
    return lower,upper
# hamming distance
def hammingD(x:int,y:int)->int:
    xor=x^y
    dist=0
    while xor:
        xor&=xor-1
        dist+=1
    return dist  
# innerproduct. np.dot(.conj(),) is much faster than np.vdot(,)
def inner_prod(u:np.ndarray,v:np.ndarray):
    return np.dot(u.conj(),v)
def Norm2(v:np.ndarray):
    return np.dot(v.conj(),v).real