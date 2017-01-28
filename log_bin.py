'''
################################################################################
# James Clough 2015
# jrc309@ic.ac.uk
# 2016 Complexity & Networks course
#
# log_bin.py v1.3
# 14/01/2016
#
# Usage instructions for log binning:
# 
# to import the log_bin functions:
# from log_bin import *
# 
# to use the log_bin function:
# arguments:
# data            - list - the numbers we want to get a probability distribution of
# bin_start       - float - the left edge of the first bin
# first_bin_width - float - the width of the first bin
# a               - float - growth factor - each bin is a times larger than the bin to its left
# datatype        - string - either 'float' or 'integer' so we know how to normalise bin widths
#
# returns:
# centres - list - the centre of each bin (in log space - ie. geometric mean)
# counts  - list - the probability of data being in each 
#
#
# Examples - available below
#
# - test_float() - Generates some power-law distributed real numbers and log-
#                  bins them. The blue points are the original numbers - the red
#                  line is the log-binning - we can see how useful it is because
#                  we get a few more orders of mangnitude of data
#
# - test_int()   - As above, but with integers
#
# - test_too_low_a() - Shows a plot where the growth factor, a, is too low
#                      and there is a kink in the line due to there being too
#                      few points in the bin
#
# Notes:
#    
# - The data will be binned with the width of bins growing exponentially so that
# even when there is less than one expected point per value we can still 
# estimate the probability distribution.
#
# - It matters whether your data are real numbers, or integers as this will change
# how we should normalise the height of each bin - use the 'datatype' argument
# to tell the function whether you are using integers or real numbers
# This is not asking whether the data you have provided are Python integers, or floats
# but rather whether you want to treat the data as integers or floats for the 
# purpose of the log binning.
#
# - Make sure to use a sensible value for 'a', which is how much larger each bin
# is than the bin to its left. If a is too small then you will have too many
# bins and too few points per bin. If it is too large then there won't be
# many bins.
#
# - If you want to have negative values on the x-axis then something will break
# when finding the centre of the bin that contains 0 - if you need to do this
# you will need a workaround as log-binning negative data is not well defined.
# 
# - The solution implemented here is not the most efficient one possible.
# possible. Feel free to tinker with this code to make it better.
#
# - There may be some bugs here - please email me if you find one.
################################################################################
'''

################################################################################
import numpy as np
import random
################################################################################
# BINNING FUNCTIONS
################################################################################
# frequency
#
# arguments:
# data - list of values
#
# returns:
# values - list of values
# frequency - list of frequencies of values
################################################################################
def frequency(data):
    from collections import Counter
    c = Counter(data)
    return np.array([float(x) for x in list(c.keys())]), np.array([float(x) for x in list(c.values())])
    
################################################################################
# lin_bin
# linear binning of data
#
# arguments:
# data     - list - the numbers we want to get the probability distribution of
# num_bins - integer - the number of bins to use
#
# returns:
# bin_centres   - list - the centres of each bin (arithmetic mean)
# normed_counts - list - the probability of data being in each bin
################################################################################
def lin_bin(data, num_bins):
    data = np.array(data)
    num_datapoints = float(len(data))
    counts, bin_edges = np.histogram(data, num_bins)
    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
    normed_counts = counts/num_datapoints
    return bin_centres, normed_counts

################################################################################
# log_bin
# log bin data of either integers, or real numbers
#
# arguments:
# data            - list - the numbers we want to get a probability distribution of
# bin_start       - float - the left edge of the first bin
# first_bin_width - float - the width of the first bin
# a               - float - growth factor - each bin is a times larger than the bin to its left
# datatype        - string - either 'float' or 'integer' so we know how to normalise bin widths
# drop_zeros      - boolean - if True, zeros are not counted in the normalisation of the probability distribution
#
# returns:
# centres - list - the centre of each bin (in log space - ie. geometric mean)
# counts  - list - the probability of data being in each bin
################################################################################
def log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
    # check datatype is valid
    valid_datatypes = ('float', 'integer')
    if datatype not in valid_datatypes:
        print('datatype must be either "float" or "integer"')
        exit(0)
        
    # ensure data is numpy array of floats
    if drop_zeros:
        data = np.array([float(x) for x in data if x!=0])
    else:
        data = np.array([float(x) for x in data])
            
    num_datapoints = len(data)
    min_x, max_x = min(data), max(data)
    
    # create array of the edges of the bins beginning with the left edge of the
    # leftmost bin, and ending with the right edge of the rightmost
    bin_width = first_bin_width
    bins = [bin_start]
    new_edge = bin_start
    while new_edge <= max_x:
        last_edge = new_edge
        new_edge = last_edge + bin_width
        bins.append(new_edge)
        bin_width *= a
    
    # find how many datapoints are in each bin
    # counts[i] is how many points are there in the bin whose left edge is bins[i]
    indices = np.digitize(data, bins[1:])
    counts = [0. for x in bins[1:]]
    for i in indices:
        counts[i] += 1./num_datapoints
        
    # normalise number of datapoints by the width of the bin
    # how we do this depends on whether we are binning integers or real numbers
    # by width - we mean the amount of possible values that can fall in that bin
    
    # we want to give a 'centre' of the bin to plot its height from
    # if the data is approximately power law distributed then the most
    # representative would be the geometric mean, since in it is in the middle
    # in log space   
    
    bin_indices = list(range(len(bins)-1)) 
    
    if datatype == 'float':
        widths = [bins[i+1] - bins[i] for i in bin_indices]
        centres = [np.sqrt(bins[i+1] * bins[i]) for i in bin_indices]
    else:
        widths = [np.ceil(bins[i+1]) - np.ceil(bins[i]) for i in bin_indices]
        integers_in_each_bin = [list(range(int(np.ceil(bins[i])), int(np.ceil(bins[i+1])))) for i in bin_indices]
        centres = [geometric_mean(x) for x in integers_in_each_bin]
            
    widths = np.array(widths)
    counts /= widths
    
    # print out some values to help with debugging
    if debug_mode:
        print('DATA - %s' % data[:10])
        print('BINS - %s' % bins[:10])
        print('INDICES - %s' % indices[:10])
        print('COUNTS - %s' % counts[:10])
        print('WIDTHS - %s' % widths[:10])
        try:
            print('INTEGERS IN EACH BIN - %s' % integers_in_each_bin[:10])
        except:
            pass
        print('CENTRES - %s' % centres[:10])
        
    return centres, counts

# returns the geometric mean of a list
def geometric_mean(x):
    s = len(x)
    y = [np.log(z) for z in x]
    z = sum(y)/s
    return np.exp(z)

################################################################################
# EXAMPLES
################################################################################
   
# generate some test data that is power law distributed
def generate_power_law_data(gamma, N):
    x = np.random.random(N)
    y = 1./(1-x)
    y = y**(1./gamma)
    y.sort()
    return y

# log bin some data
def test_float(N=100000., a=1.5):
    import matplotlib.pyplot as plt
    x = generate_power_law_data(1, N)
    vals, counts = lin_bin(x, int(max(x)))
    b, c = log_bin(x, 1., 1.5, a, debug_mode=True)
    plt.loglog(vals, counts, 'bx')
    plt.loglog(b, c, 'r-')
    plt.show() 
    
# now with integers
def test_int(N=100000., a=1.5):
    import matplotlib.pyplot as plt
    x = generate_power_law_data(1, N)
    x = np.array([int(z) for z in x])
    vals, counts = frequency(x)
    b, c = log_bin(x, 1., 1.5, a, 'integer', debug_mode=True)
    plt.loglog(vals, counts/N, 'bx')
    plt.loglog(b, c, 'r-')
    plt.show()    
      
# log bin some data - the very low growth factor gives a kink in the line
# this happens when there are too few points in each bin
def test_too_low_a():
    test_float(a=1.01)  
    
def test_normalisation(N=10000.):
    import matplotlib.pyplot as plt
    x = generate_power_law_data(1, N)
    x = np.concatenate([x, [0. for z in range(int(N))]])
    vals, counts = lin_bin(x, int(max(x)))
    b, c = log_bin(x, 1.0, 1.5, 1.5, 'float', drop_zeros=True)
    plt.loglog(vals, counts, 'bx')
    plt.loglog(b, c, 'r-')
    plt.show()
    
def main():
    print(__doc__)
    
if __name__ == "__main__":
    main()
