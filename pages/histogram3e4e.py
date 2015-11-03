""" Fast histogram """

from numpy import array, zeros, asarray, sort, int32, digitize, bincount,\
    concatenate,ones, atleast_1d, iterable, linspace, diff,log10,around,\
    arange, apply_along_axis, hsplit, argsort, sum
from scipy import weave
from typed_array_converter import converters

def histogram(a, bins=10, range=None, normed=False, weights=None, axis=None, strategy=None):
    """histogram(a, bins=10, range=None, normed=False, weights=None, axis=None) 
                                                                   -> H, dict
    
    Return the distribution of sample.
    
    :Parameters:
      - `a` : Array sample.
      - `bins` : Number of bins, or an array of bin edges, in which case the 
                range is not used.
      - `range` : Lower and upper bin edges, default: [min, max].
      - `normed` :Boolean, if False, return the number of samples in each bin,
                if True, return the density.  
      - `weights` : Sample weights. The weights are normed only if normed is 
                True. Should weights.sum() not equal len(a), the total bin count 
                will not be equal to the number of samples.
      - `axis` : Specifies the dimension along which the histogram is computed. 
                Defaults to None, which aggregates the entire sample array.
      - `strategy` : Histogramming method (binsize, searchsorted or digitize).
    
    :Return:
      - `H` : The number of samples in each bin. 
        If normed is True, H is a frequency distribution.
      - dict{ 'edges':      The bin edges, including the rightmost edge.
        'upper':      Upper outliers.
        'lower':      Lower outliers.
        'bincenters': Center of bins. 
        'strategy': the histogramming method employed.}
    
    :Examples:
      >>> x = random.rand(100,10)
      >>> H, D = histogram(x, bins=10, range=[0,1], normed=True)
      >>> H2, D = histogram(x, bins=10, range=[0,1], normed=True, axis=0)
    
    :SeeAlso: histogramnd
    """
    weighted = weights is not None
    
    a = asarray(a)
    if axis is None:
        a = atleast_1d(a.ravel())
        if weighted:
            weights = atleast_1d(weights.ravel())
        axis = 0 
        
    # Bin edges.   
    if not iterable(bins):
        if range is None:
            range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        edges = linspace(mn, mx, bins+1, endpoint=True)
    else:
        edges = asarray(bins, float)
    
    nbin = len(edges)-1
    dedges = diff(edges)
    bincenters = edges[:-1] + dedges/2.
        
    # Measure of bin precision.
    decimal = int(-log10(dedges.min())+10)
    
    # Choose the fastest histogramming method
    even = (len(set(around(dedges, decimal))) == 1)
    if strategy is None:
        if even:
            strategy = 'binsize'
        else:
            if nbin > 30: # approximative threshold
                strategy = 'searchsort'
            else:
                strategy = 'digitize'
    else:
        if strategy not in ['binsize', 'digitize', 'searchsort']:
            raise 'Unknown histogramming strategy.', strategy
        if strategy == 'binsize' and not even:
            raise 'This binsize strategy cannot be used for uneven bins.'
        
    # Parameters for the even functions.
    start = float(edges[0])
    binwidth = float(dedges[0])
    
    # For the rightmost bin, we want values equal to the right 
    # edge to be counted in the last bin, and not as an outlier. 
    # Hence, we shift the last bin by a tiny amount.
    if not iterable(bins):
        binwidth += pow(10, -decimal)
        edges[-1] += pow(10, -decimal)
    
    # Looping to reduce memory usage
    block = 66600 
    slices = [slice(None)]*a.ndim
    for i in arange(0,len(a),block):
        slices[axis] = slice(i,i+block)
        at = a[slices]
        if weighted:
            at = concatenate((at, weights[slices]), axis)        
            if strategy == 'binsize':
                count = apply_along_axis(splitinmiddle,axis,at,
                    histogram_binsize_weighted,start,binwidth,nbin)               
            elif strategy == 'searchsort':
                count = apply_along_axis(splitinmiddle,axis,at, \
                        histogram_searchsort_weighted, edges)
            elif strategy == 'digitize':
                    count = apply_along_axis(splitinmiddle,axis,at,\
                        histogram_digitize,edges,decimal,normed)
        else:
            if strategy == 'binsize':
                count = apply_along_axis(histogram_binsize,axis,at,start,binwidth,nbin)
            elif strategy == 'searchsort':
                count = apply_along_axis(histogram_searchsort,axis,at,edges)
            elif strategy == 'digitize':
                count = apply_along_axis(histogram_digitize,axis,at,None,edges,
                        decimal, normed)
                    
        if i == 0:
            total = count
        else:
            total += count
        
    # Outlier count
    upper = total.take(array([-1]), axis)
    lower = total.take(array([0]), axis)
    
    # Non-outlier count
    core = a.ndim*[slice(None)]
    core[axis] = slice(1, -1)
    hist = total[core]
    
    if normed:
        normalize = lambda x: atleast_1d(x/(x*dedges).sum())
        hist = apply_along_axis(normalize, axis, hist)

    return hist, {'edges':edges, 'lower':lower, 'upper':upper, \
        'bincenters':bincenters, 'strategy':strategy}
        


def histogram_binsize(a, start, width, n):
    """histogram_even(a, start, width, n) -> histogram
    
    Return an histogram where the first bin counts the number of lower
    outliers and the last bin the number of upper outliers. Works only with 
    fixed width bins. 
    
    :Parameters:
      a : array
        Array of samples.
      start : float
        Left-most bin edge.
      width : float
        Width of the bins. All bins are considered to have the same width.
      n : int
        Number of bins. 
    
    :Return:
      H : array
        Array containing the number of elements in each bin. H[0] is the number
        of samples smaller than start and H[-1] the number of samples 
        greater than start + n*width.
    """    
    ary = asarray(a)
    
    # Create an array to hold the histogram count results, including outliers.
    count = zeros(n+2,dtype=int32)
    
    # The C++ code that actually does the histogramming.    
    code = """
           PyArrayIterObject *iter = (PyArrayIterObject*)PyArray_IterNew(py_ary);
           
           while(iter->index < iter->size)
           {
                          
               // This requires an update to weave 
               ary_data_type value = *((ary_data_type*)iter->dataptr);
               if (value>=start)
               {
                   int bin_index = (int)((value-start)/width);
               
                   //////////////////////////////////////////////////////////
                   // Bin counter increment
                   //////////////////////////////////////////////////////////
    
                   // If the value was found, increment the counter for that bin.
                   if (bin_index < n)
                   {
                       count[bin_index+1]++;
                   }
                   else {
                       count[n+1]++; }
               }
               else {
                  count[0]++;}
               PyArray_ITER_NEXT(iter);
           }
           """
    weave.inline(code, ['ary', 'start', 'width','n', 'count'], 
                 type_converters=converters, 
                 compiler='gcc')
                 
    return count


def histogram_binsize_weighted(a, w, start, width, n):
    """histogram_even_weighted(a, start, width, n) -> histogram
    
    Return an histogram where the first bin counts the number of lower
    outliers and the last bin the number of upper outliers. Works only with 
    fixed width bins. 
    
    :Parameters:
      a : array
        Array of samples.
      w : array
        Weights of samples.
      start : float
        Left-most bin edge.
      width : float
        Width of the bins. All bins are considered to have the same width.
      n : int
        Number of bins. 
    
    :Return:
      H : array
        Array containing the number of elements in each bin. H[0] is the number
        of samples smaller than start and H[-1] the number of samples 
        greater than start + n*width.
    """    
    ary = asarray(a)
    w = asarray(w)
    # Create an array to hold the histogram count results, including outliers.
    count = zeros(n+2,dtype=w.dtype)
    
    # The C++ code that actually does the histogramming.    
    code = """
           PyArrayIterObject *iter = (PyArrayIterObject*)PyArray_IterNew(py_ary);
           
           while(iter->index < iter->size)
           {
                          
               // This requires an update to weave 
               ary_data_type value = *((ary_data_type*)iter->dataptr);
               if (value>=start)
               {
                   int bin_index = (int)((value-start)/width);
               
                   //////////////////////////////////////////////////////////
                   // Bin counter increment
                   //////////////////////////////////////////////////////////
    
                   // If the value was found, increment the counter for that bin.
                   if (bin_index < n)
                   {
                       count[bin_index+1]+=w[iter->index];
                   }
                   else {
                       count[n+1]+=w[iter->index]; }
               }
               else {
                  count[0]+=w[iter->index];}
               PyArray_ITER_NEXT(iter);
           }
           """
    weave.inline(code, ['ary', 'w', 'start', 'width','n', 'count'], 
                 type_converters=converters, 
                 compiler='gcc')
                 
    return count
       
def histogram_searchsort(a, bins):
    n = sort(a).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    count = concatenate([[n[0]], n[1:]-n[:-1]])
    return count
    
def histogram_searchsort_weighted(a, w, bins):
    i = sort(a).searchsorted(bins)
    sw = w[argsort(a)]
    i = concatenate([i, [len(a)]])
    n = concatenate([[0],sw.cumsum()])[i]
    count = concatenate([[n[0]], n[1:]-n[:-1]])
    return count

def splitinmiddle(x, function, *args, **kwds):
    x1,x2 = hsplit(x, 2)
    return function(x1,x2,*args, **kwds)

def histogram_digitize(a, w, edges, decimal, normed):
    """Internal routine to compute the 1d weighted histogram.
    a: sample
    w: weights
    edges: bin edges
    decimal: approximation to put values lying on the rightmost edge in the last
             bin.
    weighted: Means that the weights are appended to array a. 
    Return the bin count or frequency if normed.
    """
    weighted = w is not None
    nbin = edges.shape[0]+1
    if weighted:
        count = zeros(nbin, dtype=w.dtype)
        if normed:    
            count = zeros(nbin, dtype=float)
            w = w/w.mean()
    else:
        count = zeros(nbin, dtype=int32)
            
    binindex = digitize(a, edges)
        
    # Count the number of identical indices.
    flatcount = bincount(binindex, w)
    
    # Place the count in the histogram array.
    count[:len(flatcount)] = flatcount
       
    return count
    

from numpy.testing import *
from numpy.random import rand

class test_histogram1d_functions(NumpyTestCase):
    def check_consistency(self):
        n = 100
        r = rand(n)*12-1
        bins = range(11)
        a = histogram_binsize(r, bins[0], bins[1]-bins[0], len(bins)-1)
        b = histogram_digitize(r, None, array(bins), 6, False)
        c = histogram_searchsort(r,bins)
        assert_array_equal(a,b)
        assert_array_equal(c,b)
        
class test_histogram(NumpyTestCase):
    def check_simple(self):
        n=100
        v=rand(n)
        (a,b)=histogram(v)
        #check if the sum of the bins equals the number of samples
        assert_equal(sum(a,axis=0),n)
        #check that the bin counts are evenly spaced when the data is from a linear function
        (a,b)=histogram(linspace(0,10,100))
        assert_array_equal(a,10)
        #Check the construction of the bin array
        a, b = histogram(v, bins=4, range=[.2,.8])
        assert_array_almost_equal(b['edges'],linspace(.2, .8, 5),8)
        #Check the number of outliers
        assert_equal((v<.2).sum(), b['lower'])
        assert_equal((v>.8).sum(),b['upper'])
        #Check the normalization
        bins = [0,.5,.75,1]
        a,b = histogram(v, bins, normed=True)
        assert_almost_equal((a*diff(bins)).sum(), 1)
        
    def check_axis(self):
        n,m = 100,20
        v = rand(n,m)
        a,b = histogram(v, bins=5)
        # Check dimension is reduced (axis=None).
        assert_equal(a.ndim, 1)
        #Check total number of count is equal to the number of samples.
        assert_equal(a.sum(), n*m)
        a,b = histogram(v, bins = 7, axis=0)
        # Check shape of new array is ok.
        assert(a.ndim == 2)
        assert_array_equal(a.shape,[7, m])
        # Check normalization is consistent 
        a,b = histogram(v, bins = 7, axis=0, normed=True)
        assert_array_almost_equal((a.T*diff(b['edges'])).sum(1), ones((m)),5)
        a,b = histogram(v, bins = 7, axis=1, normed=True)
        assert_array_equal(a.shape, [n,7])
        assert_array_almost_equal((a*diff(b['edges'])).sum(1), ones((n)))
        # Check results are consistent with 1d estimate
        a1, b1 = histogram(v[0,:], bins=b['edges'], normed=True)
        assert_array_almost_equal(a1, a[0,:],7)
            
    def check_weights(self):
        # Check weights = constant gives the same answer as no weights.
        v = rand(100)
        w = ones(100)*5
        a,b = histogram(v)
        na,nb = histogram(v, normed=True)
        wa,wb = histogram(v, weights=w)
        nwa,nwb = histogram(v, weights=w, normed=True)
        assert_array_equal(a*5, wa)
        assert_array_almost_equal(na, nwa,8)
        # Check weights are properly applied.
        v = linspace(0,10,10)
        w = concatenate((zeros(5), ones(5)))
        wa,wb = histogram(v, bins=linspace(0,10.01, 11),weights=w)
        assert_array_almost_equal(wa, w)
        
    def check_strategies(self):
        v = rand(100)
        ae,be = histogram(v, strategy='binsize')
        ab,bb = histogram(v, strategy='digitize')
        as,bs = histogram(v, strategy='searchsort')
        assert_array_equal(ae, ab)
        assert_array_equal(ae, as)
        
        w = rand(100)
        ae,be = histogram(v, weights=w, strategy='binsize')
        ab,bb = histogram(v, weights=w, strategy='digitize')
        as,bs = histogram(v, weights=w, strategy='searchsort')
        assert_array_almost_equal(ae, ab,8)
        assert_array_almost_equal(ae, as,8)
        
        
def compare_strategies():
    """Time each strategy in the case of even bins. 
    
    The fastest method is the C based function histogram_even, which
    however cannot be used for uneven bins. In this case, searchsort is faster 
    if there are more than 30 bins, otherwise, digitize should be used. 
    """
    from time import clock
    from numpy.random import randint
    import pylab as P
    N = 2**arange(12,24)
    B = arange(10,150,20)
    strategies = ['binsize', 'searchsort', 'digitize']
    tn = zeros((len(N), len(strategies)))
    tb = zeros((len(B), len(strategies)))
    s1 = P.subplot(211)
    s2 = P.subplot(212)
    for i,s in enumerate(strategies):
        for j,n in enumerate(N):
            r = randint(0,100, n)
            t1 = clock()
            a,d = histogram(r,30, strategy=s)
            t2 = clock()
            tn[j,i]=t2-t1
        s1.loglog(N, tn[:,i], '-', marker='o', label=s)
        
        r = randint(0,100, 1e6)
        for j,b in enumerate(B):
            t1 = clock()
            a,d = histogram(r,b, strategy=s)
            t2 = clock()
            tb[j,i]=t2-t1
        s2.plot(B, tb[:,i], '-', marker='o',label=s)
    s1.set_xlabel('Number of random variables')
    s1.set_ylabel('Computation time (s)')
    s2.set_xlabel('Bin array length')
    s2.set_ylabel('Computation time (s)')
    P.legend(loc=2)
    P.savefig('timing')
    P.close()
    
    

if __name__ == "__main__":
    NumpyTest().run()
