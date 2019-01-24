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
    return np.array(data)

##Windowing Functions 
def gaussian(t,h):
    return 1/(h*(2*pi)**0.5)*e**(-t**2/(2*h**2))

def epanechnikov(t,h):
    # note numpy.where serves to vectorize this function scalars must be converted to numpy arrays or else the function will fail
    t = np.asarray(t)
    h = np.asarray(h)
    return np.where(abs(t/h) > 5**0.5,0,(3/4)*(1-(1/5)*(t/h)**2)/5**0.5)

def rectangle(t,h):
    # note numpy.where serves to vectorize this function. scalars must be converted to numpy arrays or else the function will fail
    t = np.asarray(t)
    h = np.asarray(h)
    return np.where( abs(t) > h, 0 ,1/(2*h))

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
    return x_pts,np.array([(1/N)*np.sum(f(data-x,h)) for x in x_pts]) # I could have made this a one liner... Forgive me Guido for I have sinned

#%%
data = get_data()
which_data = 2
num_pts = 1000

#%%
# epannechnikov? I hardly know her
fig,ax1 =plt.subplots()
ax2 = ax1.twinx()
x,y = kernel_density_estimator(data[which_data],.1,num_pts=num_pts)
ax1.hist(data[which_data])
ax2.plot(x,y,color='k')
plt.show()
        
# Gaussian
fig,ax1 =plt.subplots()
ax2 = ax1.twinx()
x,y = kernel_density_estimator(data[which_data],.1,f=gaussian,num_pts=num_pts)
ax1.hist(data[which_data])
ax2.plot(x,y,color='k')
plt.show()
#%%
# 
fig,ax1 =plt.subplots()
ax2 = ax1.twinx()
x,y = kernel_density_estimator(data[which_data],.1,f=rectangle,num_pts=num_pts)
ax1.hist(data[which_data])
ax2.plot(x,y,color='k')
plt.show()