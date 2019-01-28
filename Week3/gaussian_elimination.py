#%%
import numpy as np

def gaussian_elimination(x):
    '''Takes a square matrix and returns it's inverse using gaussian elimination and back substitution'''
    eps = 1e-16
    a = x.copy()
    if a.shape[0]+1 != a.shape[1]:
        print("matrix must be augmented, instead had shape {}".format(a.shape))
        return
    n = a.shape[0] # num rows 
    for c in range(0,n): # only go up to N because we only do elimination in square part
        # put partial pivoting here 
        for r in range(c+1,n):
            piv = a[c,c] # pivot point
            entry = a[r,c] # the value we are trying to eliminate
            a[r,:] -= a[c,:]*(entry/piv) # eliminating `entry`

    for c in range(n-1,-1,-1): # now we have our upper triangular augmented matrix
        a[c,:] /= a[c,c] # making pivot point 1 and entry in augmented portion correct
        for r in range(c-1,-1,-1):
            a[r,:] -= a[c,:]*a[r,c] # eliminating each entry in the upper portion of the matrix
    return a



#%% 
A = np.arange(1,10,1,dtype=float).reshape((3,3))
A[2,2] = 0
b = np.arange(1,4,dtype=float).reshape((3,1))

aug = np.hstack((A,b))
ans = gaussian_elimination(aug)
print(ans)