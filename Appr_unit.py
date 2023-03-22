import evoapproxlib as eal
import numpy as np
import evoapproxlib as eal
import warnings
import math
import struct
import random

'''from ctypes import *
import pyximport; pyximport.install()
from ctypes import *


so_file = "/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/mul8u_1JFF.so"
my_functions = CDLL(so_file)
print(type(my_functions))


wce = e = 0
for i in range(0,2**8):
    for j in range(0,2**8):
        diff = abs(my_functions.mul(i,j) - (i*j))
        if diff > wce: wce = diff
        e += diff

print('average error magnitude (mae)',e/(2.0**(16)))
print('worst-case error magnitude (wce)',wce)'''


'''import pyximport; pyximport.install()
import add12se_4ZY

wce = e = 0
for i in range(0,2**8):
    for j in range(0,2**8):
        diff = abs(add12se_4ZY.mul(i,j) - (i*j))
        if diff > wce: wce = diff
        e += diff

print('average error magnitude (mae)',e/(2.0**(16)))
print('worst-case error magnitude (wce)',wce)'''

'''e = 0
for i in range(0, 2**8):
    for j in range(0, 2**8):
        e += abs(eal.add8u_0FP.calc(i, j) - (i+j))

print('MAE calculated', e / (2**(2*8)))
print('MAE from lib', eal.add8u_0FP.MAE)'''
'''a = eal.mul8s_1KX2.calc(16, 16)
print(a)'''
for i in range(0, 16):
    for j in range(0, 16):
        e = abs(eal.mul8u_JQQ.calc(i, j) - (i*j))
        with open('mul8u_JQQ_multiplier.txt', 'a') as f:
              f.write("%s\n" % e)
        '''with open('i-index.txt', 'a') as f:
              f.write("%s\n" % i)
        with open('j-index.txt', 'a') as f:
              f.write("%s\n" % j)'''
'''for i in range (256):
    with open('i1-index.txt', 'a') as f:
         f.write("%s\n" % i)'''

'''e = 0
for i in range(0, 2**8):
    for j in range(0, 2**8):
        e += abs(eal.mul8s_1KX2.calc(i, j) - (i*j))

print('MAE calculated', e / (2**(2*8)))
print('MAE from lib', eal.mul8s_1KX2.MAE)
print(eal.mul8s_1KX2.calc(50, 15))



import pyximport; pyximport.install()


wce = e = 0
for i in range(0,2**8):
    for j in range(0,2**8):
        diff = abs(eal.mul8s_1KX2.calc(i,j) - (i*j))
        if diff > wce: wce = diff
        e += diff

print('average error magnitude (mae)',e/(2.0**(16)))
print('worst-case error magnitude (wce)',wce)
a = 10
#this will print a in 5 bit binary
bnr = bin(a).replace('0b','')
x = bnr[::-1] #this reverses an array
while len(x) < 5:
    x += '0'
bnr = x[::-1]
print(bnr)'''
