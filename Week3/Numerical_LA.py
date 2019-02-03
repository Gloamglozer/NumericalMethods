#%%
# making some nice printing functions quick
import numpy as np
from pprint import pprint
np.set_printoptions(precision=4)

def print_dict(d):
    for k,v in d.items():
        print(k+':')
        pprint(v)
    print()
    

#%% [markdown]
## Gaussian Elimination
# I was suprised at how straightforward and simple it was to make an algorithm capable of solving large systems of 
# linear equations. I had used functions to put an augmented matrix in reduced row echelon format 
# in the past but never implemented my own Gaussian elimination. 

#%%
def gaussian_elimination(A,b):
    '''Takes a square matrix and returns it's inverse using gaussian elimination and back substitution'''
    a = np.hstack((A,b)) # using augmented matrix
    eps = 1e-16
    if a.shape[0]+1 != a.shape[1]:
        print("matrix must be augmented, instead had shape {}".format(a.shape))
        return
    n = a.shape[0] # num rows 
    # intializing permutor matrix
    P = np.eye(n)
    for c in range(0,n): # only go up to N because we only do elimination in square part

    #     #partial pivoting
        this_pivot = abs(a[c,c])
        maximum = this_pivot
        max_r = c
        for r in range(c+1,n):
            potential_pivot = abs(a[r,c])
            if potential_pivot>this_pivot:
                maximum = potential_pivot
                max_r = r
        if max_r!=c: # if we have a swap
            # construct row swap matrix
            rowswap = np.eye(n)
            rowswap[:,(max_r,c)] = rowswap[:,(c,max_r)] # swap corresponding columns to get ith permutor
            P = rowswap@P # record the fact that the columns were swapped
            a = rowswap@a # go home, turn off your spigot, everybody do the partial pivot

        for r in range(c+1,n):
            piv = a[c,c] # pivot point
            entry = a[r,c] # the value we are trying to eliminate
            a[r,:] -= a[c,:]*(entry/piv) # eliminating `entry`

    U = a.copy()[:,:-1] # Make upper triangular matrix from a at this at point in time,
    c = a.copy()[:,-1].reshape((-1,1)) # as well as the c vector (just the last column of the augmented matrix)

    x = np.empty((n,1)) # make an empty column vector of the appropriate size
    # Doing back substitution
    for i in range(n-1,-1,-1):
        x[i] = (a[i,-1]-a[i,i+1:-1]@x[i+1:,0])/a[i,i] 
        # sub = a[i,i+1:-1]@x[i+1:,0]
        # divide the value in the c minus the already solved vars in column vector 
        # matrix multiplied by the row vector formed by the non pivot entries to the right of the pivot in each column
        # divided by the pivot value in the same row 

    # Note that the rows in X have been swapped if a partial pivot has occurred, undo the swaps so we can return X with the variables
    # in the correct columns
    return {'x':x,'U':U,'c':c,'P':P}

    # Doing Gaussian Elimination on lower triangular matrix
    # for c in range(n-1,-1,-1): 
    #     a[c,:] /= a[c,c] # making pivot point 1 and entry in augmented portion correct
    #     for r in range(c-1,-1,-1):
    #         a[r,:] -= a[c,:]*a[r,c] # eliminating each entry in the upper portion of the matrix

#%% 
# Example 6
A = np.array([[1,2,1],\
               [3,8,1],\
               [0,4,1]])
b = np.array([2,12,2]).reshape((3,1))
print_dict(gaussian_elimination(A,b))

#%% 
# Example 12
A = np.array([[2,6,10],\
               [1,3,3],\
               [3,14,28]])
b = np.array([0,2,-8]).reshape((3,1))
print_dict(gaussian_elimination(A,b)) # pivot on the haters
#%% [markdown]
## LU Decomposition
# This one posed more of a challenge to me. I realized that $L^{-1}$ was built out of all of 
# the ERO matricies cumulatively left multiplying A one after another. This would mean that
# similarly $L$ was all of the *inverses* of the ERO's cumulatively *right* multiplied to an 
# identity matrix.

#%%


def my_lu(a):
    '''Takes a square matrix and returns it's L U decomposition'''
    eps = 1e-16
    if a.shape[0] != a.shape[1]:
        print("matrix must be square, instead had shape {}".format(a.shape))
        return
    n = a.shape[0] # num rows 
    # intializing permutor matrix
    I = np.eye(n)
    P = np.eye(n)
    L = np.eye(n)
    for c in range(0,n): # only go up to N because we only do elimination in square part
    #     #partial pivoting
        this_pivot = abs(a[c,c])
        maximum = this_pivot
        max_r = c
        for r in range(c+1,n):
            potential_pivot = abs(a[r,c])
            if potential_pivot>this_pivot:
                maximum = potential_pivot
                max_r = r
        if max_r!=c: # if we have a swap
            # construct row swap matrix
            rowswap = np.eye(n)
            rowswap[:,(max_r,c)] = rowswap[:,(c,max_r)] # swap corresponding columns to get ith permutor
            L = L@rowswap.T # record the fact that the columns were swapped in the L matrix
            a = rowswap@a # go home, turn off your spigot, everybody do the partial pivot

        for r in range(c+1,n):
            piv = a[c,c] # pivot point
            entry = a[r,c] # the value we are trying to eliminate
            ERO = I.copy()
            ERO_inv = I.copy()
            ERO[r,c] = -(entry/piv)
            ERO_inv[r,c] = (entry/piv) # "Undoes" the row operation represented by ERO
            a = ERO@a
            L = L@ERO_inv

    return {'L':L,'U':a}

#%% Example 6
A = np.array([[1,2,1],\
               [3,8,1],\
               [0,4,1]])
d = my_lu(A)
L = d["L"]
U = d["U"]
print_dict(d) # note that L is not lower triangular, but permuted lower triangular
print("Recomposed:")
print(L@U)

#%% Example 12
A = np.array([[2,6,10],\
               [1,3,3],\
               [3,14,28]])
d = my_lu(A)
L = d["L"]
U = d["U"]
print_dict(d)
print("Recomposed:")
print(L@U)

#%% [markdown]
## Testing NumPy Orthogonalization
#%%

# Yanked from stackoverflow :)
# https://stackoverflow.com/questions/47834140/numpy-equivalent-of-matlabs-magic
def magic(n):
  n = int(n)
  if n < 3:
    raise ValueError("Size must be at least 3")
  if n % 2 == 1:
    p = np.arange(1, n+1)
    return (n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1).astype(float)
  elif n % 4 == 0:
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    M[K] = n*n + 1 - M[K]
  else:
    p = n//2
    M = magic(p)
    M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
    M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
  return M.astype(float) # changing the output of this to be of type float to make visual comparisons easier

mag = magic(5)
mag
#%%
# Comparing Q*R for Q and R produced by Numpy function for QR decomposition to original matrix
# for magic matricies with 3,5 and 10 entries
for size in [3,5,10]:
    m = magic(size)
    q,r = np.linalg.qr(m)
    print("Original Matrix:\n{}".format(m))
    print("Recomposed from QR decomposition:\n{}".format(q@r))


#%% [markdown]
# The two are equivalent.
# Showing that column vectors of Q are orthogonal:
#%% 
q,r = np.linalg.qr(mag)
from itertools import combinations
for a,b in combinations(list(range(5)),2):
    print("Column {}: {}".format(a,q[:,a]))
    print("Column {}: {}".format(b,q[:,b]))
    print("Column {} dot Column {}: {}\n".format(a,b,q[:,a]@q[:,b]))
#%% [markdown]
## Testing NumPy Eigenvalue Solving
#%%
eig_vals,eig_vecs = np.linalg.eig(mag)
pprint(eig_vals)
pprint(eig_vecs)
#%% [markdown]
# Comparing $ A x $ and $ \lambda x $ 
#%% 
for i in range(mag.shape[0]):
    print("{}/ Value of |A*v|: {}".format(i,mag@eig_vecs[:,i]))
    print("{}/ Value of λ:     {}".format(i,eig_vals[i]*eig_vecs[:,i]))