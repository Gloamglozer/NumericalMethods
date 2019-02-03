#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
cond_S_per_cm = 1000 #
size = 100 # number of divisions used for finite difference
boundary_right = 0
boundary_left = 1


A = np.empty((size,size))

#toeplitz-izing empty matrix
for i in range(size):
    A[i,i] = 2
    if i ==size-1:
        break
    A[i,i+1] = -1
    A[i+1,i] = -1


b = np.empty((size,1))

b[0,0] = boundary_left
b[size-1,0] = boundary_right

x = np.linalg.lstsq(A,b)
plt.plot(x)


#%% [markdown]
## Blah
# Problem 1: Plot the voltage as a function of position
