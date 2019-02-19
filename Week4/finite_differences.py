#%%
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
## Problem 1: Plot the voltage as a function of position
# This was solved using the normal (non-sparse) NumPy array

#%%
def solve_for_voltage(size,boundaries,cond_S_per_cm= 1000):
    ''' 
    `size` - number of divisions used to discretize line
    `boundaries` - tuple containing voltage on bounds like `(V_left,V_right)`
    `cond_S_per_cm` - conductivity
    '''

    boundary_left ,boundary_right = boundaries

    A = np.zeros((size,size))

    #toeplitz-izing empty matrix
    for i in range(size):
        A[i,i] = 2
        if i ==size-1:
            break
        A[i,i+1] = -1
        A[i+1,i] = -1

    b = np.zeros((size,1))

    b[0,0] = boundary_left
    b[size-1,0] = boundary_right

    return np.linalg.solve(A,b) # equivalent of MATLAB backslash

x = solve_for_voltage(100,(1,0))
plt.plot(x)
#%%
    
