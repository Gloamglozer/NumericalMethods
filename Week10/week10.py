#%%
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]

### Week 10 -- Initial Value Problems

## Solving for the response of an RLC circuit with the Euler method

#%%
# Initial values

dt = 0.5
t = np.arange(0,10,dt)

I0 = 0
R,L,E = 1,1,1
di_dt = lambda t,i: (-R*i+E)/L
analytic_soln = (I0-E/R)*np.exp(-(R/L)*t)+E/R

#%% [markdown]
## Deliverable 1
### Euler's Method
# Apparently Euler's method is too naive to be offered by `scipy.integrate`, so I
# ended up throwing my own simple implementation together
# $y_{n+1}= y_n + hf(t_n, y_n) $
#%%

def euler(f,y0,time):
    dt = time[1]-time[0]
    y = [y0]
    for t in time[:-1]:
        y.append(dt*f(t,y[-1])+y[-1])
    return y

plt.plot(t,analytic_soln)
plt.plot(t,euler(di_dt,I0,t))

#%% [markdown]
### Implicit Euler
# Also too naive to be implemented by a Python library, I am understanding the implicit
# euler method as a way to get a more accurate picture of the derivative by performing
# a newton's method type non-linear solver to solve $  y_{n+1} = y_n + hf(t_{n+1}, y_{n+1})  $ 
# for $ y_{n+1} $ at each point.
#
# At the ith iteration, newtons method can be performed on $  y_n + hf(t_{n+1}, y_{n+1}) - y_{n+1}$ 
# 
# The derivative of the previous expression with respect to $y_{n+1}$ would then be
# $ h*\frac{\partial}{\partial y_{n+1}} f(t_{n+1}, y_{n+1}) - 1$
#
# The partial derivative can then be evaluated numerically with a centered difference
#
# $ \frac{\partial}{\partial y_{n+1}} f(t_{n+1}, y_{n+1})  = 
# \frac{ f(t_{n+1}, y_{n+1}+\Delta y_{n+1}/2) - f(t_{n+1}, y_{n+1}-\Delta y_{n+1}/2) }{\Delta y_{n+1}}$
#
# A logical default setting in my mind was $ \Delta y_{n+1} = 0.001*y_0$ .
# 
# So, the next iterate of $y_{n+1},y_{n+1}^{i+1} $ would then be:
#
# $y_{n+1}^{i+1} = \frac{f(t_{n+1}, y_{n+1}^{i})}{h \cdot \frac{\partial}{\partial y_{n+1}} f(t_{n+1}, y_{n+1}^{i}) -1 } + y_{n+1}^{i} $
# 
# which could be computed until a tolerance is acheived or for a certain number of iterations

#%%
def implicit_euler(f,y_0,time,dy=None,newt_iters=3):
    if dy ==None:
        dy = abs(y_0*0.001)+.0001
    dt = time[1]-time[0]
    Y = [y_0] # now using uppercase for vector of y's
    for t in time[1:]:
        # Did change in notation, so everything labeled n+1 in the latex is now just the variable name,
        # and everything labeled 'n' is labeled 'last'
        # Now perform newtons method 
        y_last = Y[-1]
        y = y_last
        print('time: {}'.format(t))
        for _ in range(newt_iters):
            df_dy = (f(t,y+dy/2)-f(t,y-dy/2))/dy

            tol = y- y_last - dt*f(t,y)  # compute difference so tolerance can instead be used
            y = -f(t,y)/(dt*df_dy-1) + y

            print(abs(tol))
        Y.append(y)

    return Y

# plt.plot(di_dt(0,np.arange(0,1,0.1)))
plt.plot(t,analytic_soln)
plt.plot(t,implicit_euler(di_dt,I0,t))

#%% [markdown]
### 4th order Runge-Kutta

#%%
numeric_soln = solve_ivp(di_dt,(0,10),np.array([I0]),t_eval=t)
plt.plot(t,analytic_soln)
plt.plot(t,numeric_soln.y[0])

#%% [markdown]
## Deliverable 2
# 
#%%
a = R/(2*L)
w0 = 1/np.sqrt(L*C)
eta = a/w0
s1 = -a + np.sqrt(a**2-w0**2)
s2 = -a - np.sqrt(a**2-w0**2)
I =    