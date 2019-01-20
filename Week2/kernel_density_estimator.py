#%%
import numpy as np
import matplotlib.pyplot as plt
pi = 3.14156
e = 2.71828
## getting data
def get_data():
    data = []
    for fname in ["data{}.txt".format(i) for i in range(1,7)]:
        with open(fname,'r') as f:
            this_data = []
            for line in f.readlines():
                this_data.append(float(line))
                assert not this_data[-1] == np.nan
            data.append(this_data)

##Windowing Functions 
def gaussian(t,h):
    return 1/(h*(2*pi)**0.5)*e**(-t**2/(2*h**2))

def epanechnikov(t,h):
    if abs(t/h) > 5**0.5:
        return 0 
    else:
        return (3/4)*(1-(1/5)*(t/h)**2)/5**0.5

def rectangle(t,h):
    if abs(t) > h:
        return 0 
    else:
        return 1/(2*h)

def kernel_density_estimator(data,h,f=None,num_pts = 200,return_f=False):
    '''
    `data`      -- 1-D vector of numpy data
    `h`         -- smoothing bandwidth
    `f`         -- windowing function
    `num_pts`   -- number of points to use to show trace
    `return_f`  -- whether or not to return a function rather than trace data
    '''
    assert h>0 , "Bandwidth must be a positive number"
    if f is None: # default function is epanechnikov
        f = epanechnikov
    N = len(data)
    x_pts = np.linspace(np.amin(data),np.amax(data),num_pts)
    return np.array([(1/N)*np.sum(f(data-x,h)) for x in x_pts]),x_pts # I could have made this a one liner... Forgive me Guido for I have sinned

#%%
data = get_data
#%%
# epannechnikov? I hardly know her
x,y = kernel_density_estimator(data[0],.01)
plt.hist(data[0],'o')
plt.plot(x,y)
        
#%%
# Gaussian
x,y = kernel_density_estimator(data[0],.01,f=gaussian)
plt.hist(data[0],'o')
plt.plot(x,y)

#%%
# 
x,y = kernel_density_estimator(data[0],.01,f=rectangle)
plt.hist(data[0],'o')
plt.plot(x,y)
