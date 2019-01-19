import numpy as np
import matplotlib.pyplot as plt

def extract_data(filename,columns=None,comment=None):
    '''`columns` is one-indexed for unenlightened matlab untermensch'''
    lines = []
    data = []

    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        if not comment is None:
            line = line.split(comment)[0] # get the part before the column
        if not line.strip() =='': 
            if columns == None: # get all of the columns if columns not specified
                data.append(list(map(float,line.split())))
            else:
                data.append([float(line.split()[column-1]) for column in columns]) #
    return np.array(data)

def bin_data(data,num_bins):
    max_data = max(data)
    min_data = min(data)

    boundaries = np.linspace(min_data,max_data,num_bins+1)
    
    #offset all boundaries except last by half to get midpoints
    midpoints = boundaries[:-1] + (boundaries[1]-boundaries[0])/2

    nums = []
    for lower,upper in zip(boundaries,boundaries[1:]):
        num_in_bin = 0
        for datum in data:
            if lower<=datum<upper:
                num_in_bin += 1
        nums.append(num_in_bin)
    return np.array(nums),midpoints 

def print_hist(num_in_bin,midpoints,num_per_char=1):
    '''Combination of both formats given in example'''
    num_divs = 10
    print("  midpoint [numpts]")
    for num, midpoint in zip(num_in_bin,midpoints):
        print("{:10.5} [{:6}]:{}".format(midpoint,num,"💦"*(num//num_per_char)))
    print(' '*19+':'+'----|'*num_divs) #divs labeled underneath, have max width 5
    print(' '*19+':'+('{:5}'*num_divs).format(*[n*num_per_char for n in range(5,5*num_divs+1,5)]))

# print(extract_data('data_w_comments.txt',[2,3],comment='#'))
dat = extract_data('data.txt',[2,3],comment='#')
second_row = dat[:,0]
plt.plot(dat[:,0],dat[:,1],'o')
plt.show()

#Functions written in this script
print_hist(*bin_data(second_row,10),5)

#matplotlib.pytplot.hist with same args
plt.hist(second_row,10)
plt.show()