import numpy as np
from scipy.sparse.linalg import qmr

A = np.arange(1,10).reshape((3,3))
A[2,2] = 0

b = np.arange(1,4).reshape((3,1))
x = np.zeros(())

x = np.linalg.inv(A)@b

qmr_x= qmr(A,b,np.zeros((3,1)))