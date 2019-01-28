#%% [markdown]
# ## Week 2
# estimate_gaussian_params

#%%
import numpy as np
import timeit

def estimate_gaussian_params(data):
    N = len(data)
    rvs = {}
    rvs['mean'] = np.sum(data)/N 
    rvs['variance'] = (1/(N-1))*(np.sum((data-rvs['mean'])**2)) # minimizes roundoff error
    rvs['std_dev'] = rvs['variance']**0.5
    rvs['skew'] = (1/N)*np.sum(((data-rvs['mean'])/rvs['std_dev'])**3)
    rvs['kurt'] =  (1/N)*np.sum(((data-rvs['mean'])/rvs['std_dev'])**4)-3 
    return rvs

def analyze_data():
    for fname in ["data{}.txt".format(i) for i in range(1,7)]:
        with open(fname,'r') as f:
            data = []
            for line in f.readlines():
                data.append(float(line))
                assert not data[-1] == np.nan
                
            print('\n'+fname+':')
            print("User Values:")
            for k,v in estimate_gaussian_params(data).items():
                print("{:10}: {:10.5}  ".format(k,v),end='')
# estimate_gaussian_params(1)
def print_params(dict):
    for k,v in sorted(dict.items()):
        print("{:>10}: {:<10.5}  ".format(k,float(v)),end='')


def check_params(mean=0,std_dev=1,repeats=1):
    for _ in range(repeats):
        data = np.random.normal(loc= mean,scale = std_dev,size = (1000,))
        theory_params ={'mean':mean,'std_dev':std_dev,'variance':std_dev**2,'skew':0,'kurt':0}
        print("\nTheoretical  :",end='')
        print_params(theory_params)
        print("\nFrom function:",end='')
        print_params(estimate_gaussian_params(data))

#%%
check_params()
#%%
check_params(mean=5,std_dev=10)
#%%
check_params(mean=100,std_dev=1)
