from numpy import *
#from numba import jit
#read the single spin at position pos, 0:up 1:down#
#@jit(nopython=True)
def onebit(num : int, pos: int)-> bool:
    return bool((num&(1<<pos))>>pos)
# the sign array of every basis under Z#
#@jit(nopython=True)
def bit_signs(pos:int, dim:int, type='+-'):
    if type=='+-':
        sign0=array([1.,-1.])
    else:
        sign0=array([0,1],dtype='int')
    npos=power(2,pos)
    sign1=repeat(sign0,npos)
    signs=tile(sign1,int(dim/npos/2))
    return signs
#read two consecutive spin begin at position 'pos', pbc applied#
#@jit(nopython=True)
def twobit_seq(num: int, pos:int, L:int)->int:
    if pos<L-1:
        return (num&(3<<pos))>>pos
    else:
        return (onebit(num,0)<<1)|(onebit(num,L-1))
#@jit(nopython=True)
#read two spin at position 'pos'=(pos0,pos1)
def twobits(num:int,pos)->int:
    bit0,bit1=onebit(num,pos[0]),onebit(num,pos[1])
    return (bit1<<1)|bit0
# the sign array of every basis under ZZ#
#@jit(nopython=True)
def twobits_signs(pos:int ,L:int, dim:int, type='+-'):
    npos=power(2,pos)
    if pos<L-1:
        if type=='+-':
            sign0=array([1.,-1.,-1.,1.])
        else:
            sign0=array([0,1,1,0],dtype='int')
        sign1=repeat(sign0,npos)
        signs=tile(sign1,int(dim/npos/4))
    else:
        if type=='+-':
            sign0=array([1.,-1])
        else:
            sign0=array([0,1],dtype='int')
        sign1=tile(sign0,int(dim/4))
        signs=concatenate((sign1,flip(sign1)))
    return signs
#flipped single spin at position pos#
#@jit(nopython=True)
def flipped(num: int, pos: int)-> int:
     return num^(1<<pos)
# flipped two consecutive spins begin at position 'pos', pbc applied#
#@jit(nopython=True)
def twoflipped_seq(num: int, pos: int, L: int)->int:
    if pos<L-1:
        return num^(3<<pos)
    else:
        return (num^1)^(1<<(L-1))
#@jit(nopython=True)
#flipped two spins begin at position 'pos'=(pos0,pos1)
def twoflipped(num:int,pos)->int:
    return num^(1<<pos[0])^(1<<pos[1])
#@jit(nopython=True)
def overlap(state1, state2)->float:
    return (vdot(state1,state2)).real
#@jit(nopython=True)
def direct_product(state1, state2):
    return kron(state1,state2)