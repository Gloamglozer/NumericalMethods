#%%
pi = 3.14156
e = 2.71828
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

def kernel_density_estimator(data,h,num_pts = 200,return_function=False):
    assert h>0 , "Bandwidth must be a positive number"

