import numpy as np
# read the bit of the digit 'dig' of the string num in binary representation
def onebit(num, dig:int):
    return (num >> dig) & 1

def twobits(num, dig:tuple):
    d0, d1 = sorted(dig)
    shifted = num >> d0
    bits0 = shifted & 1
    shifted >>=(d1-d0-1)
    bits1 = shifted & 2
    return bits1 | bits0
    
#read the sign of every basis at position 'pos' according to spin Z, 0:Z=1, 1:Z=-1
def onebit_sign(basis, dig:int):
    bits = (basis >> dig) & 1
    return 1. - 2. * bits

# sign of every basis at two position correlated dig=(dig0,dig1)
def twobits_sign(basis, dig:tuple):
    xorbits = ((basis >> dig[0])^(basis >> dig[1])) & 1
    return 1. - 2. * xorbits

## the most powerful improvement!!!  Pairing the basis according to a qubit flip at position dig
# , breaking the indices 0,1,...,2^L-1 into two subarrays 
def flipp_pair(basis, dig:int, control=None):
    mask1 = ((basis >> dig) & 1).astype(bool)
    mask0 = ~ mask1
    if isinstance(control, int):
        cmask = ((basis >> control) & 1).astype(bool)
        mask0 = mask0 & cmask
        mask1 = mask1 & cmask
    return basis[mask0], basis[mask1]

def flipp_twopairs(basis, dig:tuple):
    bits = twobits(basis, dig)
    mask0 = (bits==0)
    mask1 = (bits==1)
    mask2 = (bits==2)
    mask3 = (bits==3)
    lower = np.concatenate((basis[mask0],basis[mask1]))
    upper = np.concatenate((basis[mask3],basis[mask2]))
    return lower, upper

def flipp_twopairs2(basis, dig:tuple):
    bits = twobits(basis, dig)
    lower = np.flatnonzero((bits==0)|(bits==1))
    upper = np.flatnonzero((bits==3)|(bits==2))
    return lower, upper
# hamming distance
def hammingD(x:int, y:int)->int:
    xor = x^y
    dist = 0
    while xor:
        xor &= xor - 1
        dist += 1
    return dist  
# innerproduct. np.dot(.conj(),) is much faster than np.vdot(,)
def inner_prod(u:np.ndarray, v:np.ndarray):
    return np.dot(u.conj(), v)
def Norm2(v:np.ndarray):
    return np.dot(v.conj(), v).real

if __name__ == "__main__":
    rng = np.random.default_rng()
    basis = np.arange(1<<5)
    arr = rng.integers(5, size=2)
    dig = arr[0], arr[1]
    print(arr)
    print(flipp_twopairs(basis, dig))
    print(flipp_twopairs2(basis, dig))
    