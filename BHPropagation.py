#!/cygdrive/c/Python27/python

import sys
import os
import optparse
from copy import copy

import numpy as np
import pandas as pd

import os.path
import datetime, time

import glob

import pickle 

import matplotlib.pyplot as plt      
import random as rn
#from bisect import bisect_left
import bisect
import scipy.stats as stats
from scipy.stats import linregress
import textwrap





# Copyright 2026 Hrvoje J Hrgov\"ci\'c

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




global FreshProb
global probtracker


bPrintDesc = False
if bPrintDesc:
    print("""

These programs construct a generic graph with continuous, integer (and also Z-token) amplitudes. See paper for clarification.

We will then proceed to define heat-propagation (i.e. random walk), modpool heat-prop, B-H propagation, modpool B-H prop
(and eventually, boson and fermion versions, with the former employing a bath of Z-tokens, which ensure that no
outgoing edge has a value other than -1, 0, or 1 -- i.e. it satisfies a Pauli Exclusion principle). We can also (in the
case of bosons) set the maximum particle absolute amplitude to some number > 1.

To avoid clutter, we have omitted the implementation of "naive regularization" which consists merely of spewing 
particle-antiparticle pairs whenever doing so at a given node would reduce the overall outgoing particle number. 

Note that in this program we use Amplitude to define the count at a given arc of a node. In the paper we reserve amplitude
for the sum of the particle counts at all the arcs leading into (or out of) a node.

Also note that the graphs here may be randomly generated to satisfy some basic properties about their respective
adjacency matrices and their average number of neighbors and so forth. They will be suitable for
generating waves only if their random walk distributions are Gaussian (with the same k). 
So while the D-dimensional rectangular lattices satisfy that criterion (if they're large enough for 
particle densities to be quasi-continuous) the other graphs might not even have a well-defined Euclidean dimension. 
Even so, we can still generate random walks and Brownian-Huygens propagation (and variations thereof) on their 
nodes, and see how the overall particle number grows with time.

At any point, any generated graph, call it Xgraph, can be saved to some pickle file named "Xpickle.pkl" by invoking the 
command SaveGraph(Xgraph, "Xpickle.pkl"), and then analyzed at some later point for its adjacency matrix, particle distribution, etc.

The code here allows for a given imput configuration to be generated many times (as for a Monte Carlo simulation)
in order to verify that the expected number of particles matches the continuous case (or the ModPool case -- again,
see the paper for clarification) for a high enough number of runs.
 In the special case of a 2-d graph of length 4 in either dimension we can use
the --print option to generate helpful 4x4 diagrams of the evolution of the graph. This proved very useful for debugging.
With minor modification, one can use this approach to observe any 4-by-4 quadrant of a graph (or a 2-D 4x4 slice of
some larger graph). Also, various  modules beginning with "Export" are used to obtain matrix representations of
2-D graphs, which also proved very handy. Testing any set of dynamics was done by way  of integrating the squared
error w.r.t. the continuous (floating-point) case corresponding to any initial configuration and number of time steps.
(This is is refered to as chi-squared in the code.)

The  ModPool version was also computed along with any fermi/bose attempt, so that the convergence  of the Fermi/Bose dynamics can be compared to the
more straightforward ModPool version.

Here is a sample script for observing chi-squared error inside ipython:

%run  thisfile.py --dim 2 --length 8 --steps 4 --dynamics bose  --seed 579289 --runs 800  --boundaryconditions InitializeDiracDelta

The graph in this question is a 2-dimensional lattice (of length 8, so 8 x 8 nodes in all). The initial conditions are "InitializeDiracDelta" which you can see corresponds to a hodotic solution (see the paper) centered at the middle of the graphs. The run count is 800, so the routine generates 800 Monte Carlo runs, and each time exports the amplitudes of a graph into a matrix (this is done automatically for 2-d graphs) and the count after 4 steps is incremented.

The continuous case is also computed, as is the modpool, so that after the requisite number of steps the sumsquared error (or chi-sq) at each node.

A second mode of usage is to set the number of MonteCarlo runs to 1, and iterate the graph over many time steps and
observe the growth in the absolute number of particles, which is linear with time in the case of pure Brownian-Huygens
Propagation, proportional to the square root of time in the case of ModPool (or its variations, as discussed in the paper),
or else, bounded (barring the usual momentary spikes that can be brushed off as another quantum fluctuation). Search
on linregress(tarr, absarr) to see how this was done.

 %run  thisfile.py --dim 2 --length 8 --steps 1000 --dynamics bose  --seed 579289 --runs 1

Note that 'bose' dynamics simply means a dynamics for which there is a generalized Pauli Exclusion Principle
that mandates that the particle count never exceeds M (where M is given by --limit, or if that is absent, is 1).
In other words, the following switches: --dynamics bose --limit 1 
will give the same dynamics as for Fermi particles, in the sense that both obey the (standard) Pauli Exclusion Principle. 






""")

CUTOFF = 0








def MonteCarlo(D, R, nruns=15, nstep=5):
    # generate a bunch (of size nruns) of random walks, each  of of length nstep
    nsuccess = 0
    p = np.ones((D,))/float(D)
    
    sumvar = 0
    NSucc = 0
    for i in range(nruns):
        x = np.zeros((D,))
        pvec = np.ones((D,))/float(D)
        
        if D > 1:
            randvec = rn.multinomial(nstep, p) # rn.normal(size=(D,nstep))/nstep
        else:
            randvec = [nstep]
        for ii,ir in enumerate(randvec):
            myres = rn.multinomial(ir, [0.5, 0.5])
            x[ii] = myres[0] - myres[1]
            #print(x, randvec, myres)
        sumsq = np.sum(x * x)
        sumvar += sumsq
        if sumsq <= R * R:
            NSucc += 1

    print("%f %f %f" % (NSucc/float(nruns), sumvar/float(nruns), np.sqrt(sumvar/float(nruns))))

    return NSucc / float(nruns)
# MonteCarlo(2,1.666,1,1)

#
#D=1; k=1; R=1; t=1; NSteps=100
# PDEGauss(D,k,R,t,NSteps= 100)

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



# old routine -- uses arrays, not generic graphs
# generate heat equation (starting from delta function init. conditions) on a D-dimensional rectuangular grid and verify that the
# behavior has the same  decay as in the continuous  case (i.e. with the standard heat kernel where r has been set to 0)
def PDEGauss(D,R,k=1,t=1,NSteps=100):
    # remember sigma^2 = 2kt

    orighist = []
    
    if D == 1:
        grid = np.zeros((NSteps*2 + 10))
        midpt = grid.shape[0]//2
        grid[midpt] = 1.0


        
        for istep in range(1,NSteps+1):
            chunk = copy(grid[(midpt-istep):(midpt+istep)])
            grid[(midpt-1-istep):(midpt-1+istep)] = grid[(midpt-1-istep):(midpt-1+istep)] + 0.5 * chunk
            grid[(midpt+1-istep):(midpt+1+istep)] = grid[(midpt+1-istep):(midpt+1+istep)] + 0.5 * chunk
            
            grid[istep % 2::2] = 0
            if (istep + midpt) % 2 != 0:
                orighist.append(grid[midpt])

        sumvar = 0
        for i in range(grid.shape[0]):
            sumvar += (i-midpt) * (i-midpt) * grid[i]
        print("var %f sd  %f" % (sumvar, np.sqrt(sumvar)))
        
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))

        return orighist, np.sum( grid[lo : hi ]), np.sum(grid)

    if D == 2:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt] = 1.0

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    if (istep+i+j) % 2 == 1:
                        grid[i, j] = 0
                    else:
                        grid[i, j] = grid[i, j] + 1/4.0*(gridcopy[i-1,j] + gridcopy[i+1,j] + gridcopy[i,j-1] + gridcopy[i,j+1])

            if (istep + D*midpt) % 2 == 0:
                orighist.append(grid[midpt,midpt])



            
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))


        sumvar = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                sumvar += ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt)) * grid[i,j]

        retval = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt)) <= R * R:
                    retval += grid[i,j]
        print("var %f sd  %f %f" % (sumvar, np.sqrt(sumvar), np.sum(np.sum(grid))))
        return orighist, np.sum( grid[lo:hi, lo:hi] )

    

    if D == 3:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt, midpt] = 1.0

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    for k in range(midpt-istep,midpt+istep+1):
                        if (istep+i+j+k) % 2 == 1:
                            grid[i, j, k] = 0
                        else:
                            grid[i, j, k] = grid[i, j, k] + 1/6.0*(gridcopy[i-1,j,k] + gridcopy[i+1,j,k] + gridcopy[i,j-1,k] + gridcopy[i,j+1,k] + gridcopy[i,j,k-1] + gridcopy[i,j,k+1])

            if (istep + D*midpt) % 2 == 0:
                orighist.append(grid[midpt,midpt,midpt])
                print(grid[midpt,midpt,midpt])

            
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))


        sumvar = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    sumvar += ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt) + (k-midpt) * (k-midpt)) * grid[i,j,k]

        retval = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    if ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt) + (k-midpt) * (k-midpt)) <= R * R:
                        retval += grid[i,j,k]
        print("var %f sd  %f %f" % (sumvar, np.sqrt(sumvar), np.sum(np.sum(np.sum(grid)))))
        return orighist, np.sum( grid[lo:hi, lo:hi, lo:hi] )


    if D == 4:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt, midpt, midpt] = 1.0
        orighist = []

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    for k in range(midpt-istep,midpt+istep+1):
                        for m in range(midpt-istep,midpt+istep+1):
                            if (istep+i+j+k+m) % 2 == 1:
                                grid[i, j, k, m] = 0
                            else:
                                grid[i, j, k, m] = grid[i, j, k, m] + 1/8.0*(gridcopy[i-1,j,k,m] + gridcopy[i+1,j,k,m] + gridcopy[i,j-1,k,m] + gridcopy[i,j+1,k,m] + gridcopy[i,j,k-1,m] + gridcopy[i,j,k+1,m] + gridcopy[i,j,k,m-1] + gridcopy[i,j,k,m+1])

            if (istep + D*midpt) % 2 == 0:
                orighist.append(grid[midpt,midpt,midpt,midpt])
                print(grid[midpt,midpt,midpt,midpt])

            
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))


        sumvar = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    for m in range(grid.shape[3]):
                        sumvar += ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt) + (k-midpt) * (k-midpt) + (m-midpt) * (m-midpt)) * grid[i,j,k,m]

        retval = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    for m in range(grid.shape[3]):
                        if ((i-midpt) * (i-midpt) + (j-midpt) * (j-midpt) + (k-midpt) * (k-midpt) + (m-midpt) * (m-midpt)) <= R * R:
                            retval += grid[i,j,k,m]
        print("var %f sd  %f %f" % (sumvar, np.sqrt(sumvar), np.sum(np.sum(np.sum(np.sum(grid))))))
        return orighist, retval, np.sum( grid[lo:hi, lo:hi, lo:hi, lo:hi] )






def PDETouch(D,R,k=1,t=1,NSteps=100):
    # this is 1 if neighbor node is 1; when started with a Dirac delta, it measures for each N
    # the expanding topological footprint consisting of all nodes reacheble with N; i.e. in an array  
    # it is the D-dimensional diamond, in some other D-dimensional structure, it will be what amounts to
    # a D-dimensional blob (or so I'd guess), in other cases, the expanding sum of the graph will imply
    # some weird non-integer growth.

    orighist = []
    
    if D == 1:
        grid = np.zeros((NSteps*2 + 10))
        midpt = grid.shape[0]//2
        grid[midpt] = 1.0


        ngrid = grid.shape[0]
        for istep in range(1,NSteps+1):
            for i in range(ngrid):
                if grid[(i+1) % ngrid] != 0 or  grid[i-1] != 0:
                    grid[i] = 1.0
            print(np.sum(grid))

        
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))


        return orighist, np.sum( grid[lo : hi ]), np.sum(grid)

    if D == 2:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt] = 1.0

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    if (istep+i+j) % 2 == 1:
                        grid[i, j] = 0
                    else:
                        if grid[i, j] + 1/4.0*(gridcopy[i-1,j] + gridcopy[i+1,j] + gridcopy[i,j-1] + gridcopy[i,j+1]) > 0:
                            grid[i, j] = 1.0
            print(np.sum(np.sum(grid)))
            
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))

        return orighist, np.sum( grid[lo:hi, lo:hi] )

    

    if D == 3:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt, midpt] = 1.0

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    for k in range(midpt-istep,midpt+istep+1):
                        if (istep+i+j+k) % 2 == 1:
                            grid[i, j, k] = 0
                        else:
                            if grid[i, j, k] + 1/6.0*(gridcopy[i-1,j,k] + gridcopy[i+1,j,k] + gridcopy[i,j-1,k] + gridcopy[i,j+1,k] + gridcopy[i,j,k-1] + gridcopy[i,j,k+1]) > 0:
                                grid[i, j, k] = 1.0
            print(np.sum(np.sum(np.sum(grid))))

            
        Radj = np.round(R*np.sqrt(NSteps)) / np.sqrt(2*k*t)
        lo = int(np.max([0, midpt-Radj]))
        hi = int(np.min([grid.shape[0], midpt+Radj]))

        return orighist, np.sum( grid[lo:hi, lo:hi, lo:hi] )


    if D == 4:
        grid = np.zeros((NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6, NSteps*2 + 6))
        midpt = grid.shape[0]//2
        grid[midpt, midpt, midpt, midpt] = 1.0
        orighist = []

        for istep in range(1,NSteps+1):
            gridcopy = copy(grid)
            for i in range(midpt-istep,midpt+istep+1):
                for j in range(midpt-istep,midpt+istep+1):
                    for k in range(midpt-istep,midpt+istep+1):
                        for m in range(midpt-istep,midpt+istep+1):
                            if (istep+i+j+k+m) % 2 == 1:
                                grid[i, j, k, m] = 0
                            else:
                                if grid[i, j, k, m] + 1/8.0*(gridcopy[i-1,j,k,m] + gridcopy[i+1,j,k,m] + gridcopy[i,j-1,k,m] + gridcopy[i,j+1,k,m] + gridcopy[i,j,k-1,m] + gridcopy[i,j,k+1,m] + gridcopy[i,j,k,m-1] + gridcopy[i,j,k,m+1]) > 0:
                                    grid[i, j, k, m] = 1.0
            print(np.sum(np.sum(np.sum(np.sum(grid)))))
        #print("var %f sd  %f %f" % (sumvar, np.sqrt(sumvar), np.sum(np.sum(np.sum(np.sum(grid))))))
        return orighist, retval, np.sum( grid[lo:hi, lo:hi, lo:hi, lo:hi] )






def Krnl(t,r,k,D):
    multip = 1.0/ (4 * math.pi * k * t)**(D/2.0) 
    return multip * np.exp(-r*r/(4*k*t))

def S_nminus1(nminus1):
    if nminus1 == 0:
        return 2 # r**0
    if nminus1 == 1:
        return 2 * math.pi # * r**1
    if nminus1 == 2:
        return 4 * math.pi # * r ** 2

    n = nminus1n + 1.0
    return 2 * (math.pi ** (n/2)) / math.gamma(n/2)


def Integ1(R,k,t,delta=0.001):
    D = 1
    multip = 1.0/ (4 * math.pi * k * t)**(D/2.0) 
    integg = 0
    for i in np.arange(-R, R+delta, delta):
        integg += np.exp(-i * i / 4.0 / t)
    
    return multip * integg * delta


def Integ2(R,k,t,delta=0.01):
    D = 2
    multip = 1.0/ (4 * math.pi * k * t)**(D/2.0) 
    integg = 0
    for i in np.arange(-R, R+delta, delta):
        for j in np.arange(-R, R+delta, delta):
            if i*i + j*j <= R * R:
                integg += np.exp(-(i * i + j * j) / 4.0 / t)
    
    return multip * integg * delta * delta


def IntegKrnl(D, R, k, t):
    multip = 1.0/ (4 * math.pi * k * t)**(D/2) / S_nminus1(D-1) 
    a = 4 * k * t
    volelementfac = S_nminus1(D-1)
    if D == 1:
        return np.sqrt(a * math.pi) * 0.5 * math.erfc(R/np.sqrt(a))
    if D == 2:
        return -volelementfac * a/2.0 * (np.exp(-R*R/a) - 1.0)

# in D dimensions the surface element is S_nminus1(D-1) r**(D-1) * dr
# so u = r**(D-2) while dv = -r * Krn(t,r,k,D) / 2 / k / t  (which implies that v = Krnl)
# i.e.  -2kt * integ( u * dv ) = -2kt * ( u(R)v(R)  - 0   - ingteg( (D-2) * r**(D-3) )  )


def ReadParams():

    p = optparse.OptionParser()
        
    p.add_option("--tutorial", default=False,
                    action="store_true", dest='tutorial', 
                    help="run in tutorial mode")
        
    p.add_option("-d", "--dimension", "--dim", default=0,
                    action="store", type="int", dest='dimension',
                    help="if this and the --length option are positive, the graph is a D-dimansional array with the specified length")

    p.add_option("--picklefile", default="",
                    action="store", dest='picklefile', type='string',
                    help="saves graph so that (if one modifies the code) it can be loaded")

    p.add_option("--loadfrompkl", default="",
                    action="store", dest='loadpicklefile', type='string',
                    help="if not empty, graph will be loaded from the pickle file")


    p.add_option("--loadfromadjacencyfile", default="",
                    action="store", dest='loadadjacencyfile', type='string',
                    help="if not empty, graph will be loaded from the pickle file")

    p.add_option("--seed", default=84848484,
                    action="store", dest='seed', type='int',
                    help="rand number seed ")

    p.add_option("--prob", "--probability", "-p", default=0,
                    action="store", dest='prob', type='float',
                    help="if length of square adjacency matrix given by --length is N, then each component above the diag is a 1 with prob p but each  node will have at least --minneighbor neighbors and less than --maxneighbor")


    p.add_option("--initprob", "--initprobability", default=0,
                    action="store", dest='initprob', type='float',
                    help="we will generate a 1 (or else a -1) for an neighbor of a node with prob p and will ensure that the initial sum of the amplitudes integrates to zero. Feel free to modify code if that's too restrictive. ")

    p.add_option("--initamp", "--initamplitude", default=0.001,
                    action="store", dest='initprob', type='float',
                    help="if positive, then p percent of initial arcs will have a 1 (times init amp); if negative then p will have 1 (times amp) and p wil have -1, and if they cancel, they cancel")


    p.add_option("--init", "--initialconditions", "--boundaryconditions", default="",
                    action="store", dest='initialconditions', type='string',
                    help="if positive, then p percent of initial arcs will have a 1 (times init amp); if negative then p will have 1 (times amp) and p wil have -1, and if they cancel, they cancel")


    p.add_option("--length", default=-1,
                    action="store", dest='length', type='int',
                    help="if --dimension is positive, then graph will have length**D nodes; if not, it will have this many nodes and use an adjacency matrix to assign structure")


    p.add_option("--minmax", "--minmaxnbr", default='0,0',
                    action="store", dest='minmaxnbr', 
                    help="each node will be connected to randomly chosen other nodes in a (possibly unsuccessful, but not very  much so) attempt to give every node between minnbr and maxnbr neighbors; therefore, the argument in this case is a csv 2-tuple specifying the minimum and maximum number of neighbors each node should have")

    p.add_option("--steps", default=0,
                    action="store", dest='steps', type='int',
                    help="specified propagation will be executed for this many steps so as to determine how the total particle number grows")

    p.add_option("--runs","--simulations", default=10000,
                    action="store", dest='nruns', type='int',
                    help="how many times (for a single graph created only once) will the nstep simulation be run.")

    p.add_option("--dynamics", default="",
                    action="store", dest='dynamics', type='string',
                    help="can be bernoulli or purebrownian or modpool (bernouilly doesn't preserve particle count), purebrownian is just pure BHP, and modpool is BHP with min variance)")


    p.add_option("--zdynamics", default="pedantic",
                    action="store", dest='zdynamics', type='string',
                    help="in this tutorial version, this can only be pedantic, but assuming the user wants to try code another version, the switch has been left as it.")


    p.add_option("--CUTOFF", default=0.0,
                    action="store", dest='CUTOFF', type='int',
                    help="threshold for emitting particle/antiparticle pairs (naive regularization) ")

    p.add_option("--CUTOFFVOLUME", "--cutoffvolume", "--CUTOFFVOL", "--cutoffvo", default=0.0,
                    action="store", dest='CUTOFFVOLUME', type='int',
                    help="if nonzero, limits the amuont of offsetting particle/antiparticle pairs emitted by each breach of --CUTOFF")


    p.add_option("--diagonalize", default=False,
                    action="store_true", dest='diagonalize', 
                    help="only works if --dim option is activated, this allows an extra 2 neighbors (for a total of 2(D+1) on each node; node that if D is even the result is NOT systolic/bipartite anddiagonalized  result is therefore of interest.)")


    p.add_option("--print", default=False,
                    action="store_true", dest='print', 
                    help="should only be used with length 4 2-d graphs (possibly larger ones, though you'll have to modify the code to focus  on some 4x4 matrix of interest)")


    p.add_option("-m", "--limit", "--particlelimit", default=0,
                    action="store", dest='limit', type='int',
                    help="must be some integer > 0; to simulate fermions, you can iether select 'fermi' in the dynamics, or select 'bose' and also set this parameter to 1; if 'VARY' is chosen, the limit is set as low as it can be at any node (so that for a starting configuration  where abs(amplitude) is <=1, it is indistinguishable from Fermi case)")


    p.add_option("--regprob", "--regprobability", default=1.1,
                    action="store", dest='regprob', type='float',
                    help="regularization probability -- used only with ModPool or PureBHP; the smaller it is the less likely that naive regularization will be performed at a node")

    p.add_option("--initlist", default=False,
                    action="store_true", dest='initlist', 
                    help="list the available arguments for --boundaryconditions")

    
                    
    return p.parse_args()                



class growthprob_t():
    def __init__(self, M, ggraph):
        self.bActive = True
        self.M = M
        self.Reset()
        self.xarr = []
        self.NormFactor = float(np.sum( [len(inode.Neighbors) for inode in ggraph.NodeVec] ))
    
    def Reset(self, ):
        self.NUpdate = 0
        self.PreRegBreachFlow = 0 # change in Nabs before regularization
        self.RegBreachReductionFlow = 0 # reduction in Nabs due to regularization
        self.NabsIncrease = 0
        #self.RemainingBreachFlow = 0 # if regularization is only partial
        

    def UpdateXarr(self, ):
        self.xarr.append([self.PreRegBreachFlow/self.NormFactor, self.RegBreachReductionFlow/self.NormFactor, self.NabsIncrease/self.NormFactor])
        self.Reset()

    def PrintMeans(self,):
        print("PreRegBreachFlow", np.mean( np.array(probtracker.xarr)[:,0] ))
        print("RegBreachReductionFlow", np.mean( np.array(probtracker.xarr)[:,1] ))

    def GetBreach(self, xarr):
        sumabsbreach = 0
        for ix in xarr:
            if ix >= 0:
                sumabsbreach += np.max([ix - self.M, 0])
            else:
                sumabsbreach += -np.min([ix + self.M, 0]) 
        return sumabsbreach

    def UpdateModPool(self, incarr, preregoutarr, outarr):
        
        if np.sum(np.abs(incarr)) == 0:
            return

        inputbreach = self.GetBreach(incarr)
        preregbreach = self.GetBreach(preregoutarr)
        outbreach = self.GetBreach(outarr)

        self.NUpdate += 1
        self.PreRegBreachFlow += preregbreach 
        self.RegBreachReductionFlow -= outbreach - preregbreach  # outbreach - preregbreach 
        self.NabsIncrease = np.sum(np.abs(preregoutarr)) - np.sum(np.abs(incarr))
        #self.RemainingBreachFlow += outbreach


    def UpdateManageZ(self, origerr, finalerr):
        
        origsumerr = np.sum(np.abs(origerr))
        if origsumerr == 0:
            return

        finalsumerr = np.sum(np.abs(finalerr))
        self.NUpdate += 1
        self.PreRegBreachFlow += origsumerr
        self.RegBreachReductionFlow += origsumerr - finalsumerr  # outbreach - preregbreach 
        self.NabsIncrease = finalsumerr
        #self.RemainingBreachFlow += outbreach




# https://stackoverflow.com/questions/18500541/how-to-flatten-a-tuple-in-python
def flatten(data):
    """ this is a useful function, but it isn't used  here
    """
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)



class correlwatch_t():
    def __init__(self,):
        self.clear()

    def sd(self, sumx, sumx2, N):
        if N <= 1:
            return 0
        return np.sqrt( float( sumx2 - sumx * sumx / float(N) ) / float(N))

    def mu(self, sumx, N):
        if N <= 0:
            return 0
        return sumx / float(N)

    def clear(self,):
        self.x = []
        self.y = []
        self.N = 0
        self.sumx = 0
        self.sumy = 0
        self.sumxy = 0
        self.sumx2 = 0
        self.sumy2 = 0

    def Increment(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.N += 1
        self.sumx += x
        self.sumy += y
        self.sumxy += x * y
        self.sumx2 += x * x
        self.sumy2 += y * y
        

    def corrcoeff(self,):
        sigx = self.sd(self.sumx, self.sumx2, self.N)
        sigy = self.sd(self.sumy, self.sumy2, self.N)
        mux = self.mu(self.sumx, self.N)
        muy = self.mu(self.sumy, self.N)
        #import pdb; pdb.set_trace()
        return ( self.sumxy  - self.sumx * self.sumy / float(self.N) ) / (sigx * sigy) /np.sqrt(float(self.N ))/np.sqrt(self.N)
        

class dictionaryiterator_t:
    """ iterates over a dictionary of lists, so that if keys A, B, and C have lists of 3, 4, 5 elemennts each,
    you can iterate over every distinct combination
    """

    def __init__(self, basedict):
        if np.min(list(basedict.values())) < 1:
            print("error in dictionaryiterator_t() -- components of base list must be > 0")
            import pdb; pdb.set_trace()
        self.BaseDict = basedict
        self.KeyList = list(basedict.keys())
        self.KeyList.sort()
        self.N = len(basedict)
        self.Period = np.prod(list(basedict.values()))
        self.Current = copy(basedict)
        for key,val in self.Current.items():
            self.Current[key] = 0
        self.CurrentN = 0
    
    def Increment(self,):
        bDone = False
        ikey = self.N - 1
        while not(bDone):
            key = self.KeyList[ikey]
            if self.Current[key] == (self.BaseDict[key] - 1):
                self.Current[key] = 0
                ikey -= 1
                if ikey < 0:
                    bDone = True
                
            else:
                bDone = True
                self.Current[key] += 1

        self.CurrentN = self.GetCurrentN()

    def GetCurrentN(self,):
        imult = 1
        xsum = 0
        for ikey in range(len(self.BaseDict)-1,-1,-1):
            key = self.KeyList[ikey]
            xsum += self.Current[key] * imult
            imult *= self.BaseDict[key]
        return xsum

    


class node_t():
    def __init__(self, id):
        self.Id = id
        self.Neighbors = [] # here, every element of the list is of type neighbor_t
        self.FromNbr = []
        self.Amplitude = [] 
        self.ContAmplitude = []
        self.AmplitudeCtl = [] # this is the "control" i.e. the pure ModPool case which both the Fermi and Bose models must replicate at each node (when in/out arcs are summed)
        self.Scratch = [] # we calc next step here until ALL the nodes are updated, and then we this into Amplitude -- otherwise, some ainitial amplitudes get wiped out
        self.ScratchCtl = []
        self.ContScratch = [] # we calc next step here until ALL the nodes are updated, and then we this into Amplitude -- otherwise, some ainitial amplitudes get wiped out
        self.PrevAmplitude = []
        self.Parity = -1 # this is for the D-dimensional array cases, and is 0 if the sum of the components of the coordinate representation
        self.Coords = [] # only assigned in the case graph is a D-dimensional array
        self.ZAmplitude = [] 
        self.ZScratch = []


        
                              
        # of a node are even, and odd otherwise.

        

class graph_t():
    def __init__(self, ): #, NMaxNeighbors=-1):
        """
        The default construction will be an NDim+1 dimensional cube (so as to have a genus equivalent to a ball in NDim+1 dimensions)

        But this can also accommodate generic graphs if given a adjacency matrix

        """


        self.NDim = 0 # this is only used if the grid is a D-dimensional array
        self.TorLen = 0 # 32 # this is only used if the grid is a D-dimensional array

        self.NodeVec = [] # decided to use a vector instead of dictionary to hold the nodes
        self.NNode = 0 

        self.FromNbr = {} # makes iteration easier -- see code

        self.bNeedZTokens = False


        #these are used only for debugging -- get rid of them
        self.irun = 0
        self.t = 0
        


    def GetFromNodeNbr(self, ):
        
        for i in range(self.NNode):
            self.NodeVec[i].FromNbr = [-1 for i in range(len( self.NodeVec[i].Neighbors))]
            for iinbr, inbr in enumerate(self.NodeVec[i].Neighbors):

                for jjnbr, jnbr in enumerate(self.NodeVec[inbr].Neighbors):
                    if jnbr == i:
                        self.NodeVec[i].FromNbr[iinbr] = jjnbr
                        break
                        
                
        for i in range(self.NNode):  
            if np.min(self.NodeVec[i].FromNbr) < 0 or len(self.NodeVec[i].FromNbr) != len(self.NodeVec[i].Neighbors):
                print("wrong")
                import pdb; pdb.set_trace()





    def MaxAbsAmp(self,):
        maxamp = 0
        for inode in  self.NodeVec:
            maxamp = np.max([maxamp, np.max(np.abs(inode.Amplitude))])
        return maxamp



    def SumAmp(self,bPrint=True):
        myret = np.sum([np.sum(inode.Amplitude) for inode in self.NodeVec])
        if bPrint:
            print("Sumamp ", myret)
        return myret


    def SumAmpCtl(self,bPrint=True):
        myret = np.sum([np.sum(inode.AmplitudeCtl) for inode in self.NodeVec])
        if bPrint:
            print("Sumampctl ", myret)
        return myret


    def SumAbsCtl(self,bPrint=True):
        myret = np.sum([np.abs(inode.AmplitudeCtl) for inode in self.NodeVec])
        if bPrint:
            print("Sumampctl ", myret)
        return myret


    def SumAbs(self,bPrint=True):
        myret = np.sum([np.sum(np.abs(inode.Amplitude)) for inode in self.NodeVec])
        if bPrint:
            print("Sumabs ", myret)
        return myret


    def MDist(self,bPrint=True):
        retvec = []
        for inode in self.NodeVec:
            M = np.abs(np.sum(np.abs(inode.Amplitude)))
            retvec.append(M)
        retvec.sort()
        retvec.reverse()

        return np.array(retvec)[:len(retvec)//2]



    def SumAmpZ(self,bPrint=True):
        myret = np.sum([np.sum(inode.ZAmplitude) for inode in self.NodeVec])
        if bPrint:
            print("SumampZ ", myret)
        return myret



    def SumAbsZ(self,bPrint=True):
        myret = np.sum([np.sum(np.abs(inode.ZAmplitude)) for inode in self.NodeVec])
        if bPrint:
            print("SumabsZ ", myret)
        return myret


    def SumAmpCont(self,bPrint=True):
        myret = np.sum([np.sum(inode.ContAmplitude) for inode in self.NodeVec])
        if bPrint:
            print("Sumampcont ", myret)
        return myret







    def Sumx(self,bPrint=True):
        sumx = 0
        sumx2 = 0
        for inode in self.NodeVec:
            x = np.sum(inode.Amplitude)
            sumx += x
            sumx2 += x * x
        myret = sumx, sumx2
        if bPrint:
            print("Sumampx ", myret)
        return myret

    def SumxZ(self,bPrint=True):
        sumx = 0
        sumx2 = 0
        for inode in self.NodeVec:
            x = np.sum(inode.ZAmplitude)
            sumx += x
            sumx2 += x * x
        myret = sumx, sumx2
        if bPrint:
            print("Sumampx ", myret)
        return myret

    def UpdatePrev(self,):
        for inode in self.NodeVec:
            inode.PrevAmplitude = inode.Amplitude

    def Touch(self, NSteps):
        print("Touch")
        myretarr = []
        x = self.SumAmp(False)
        myretarr.append( x/float(self.NNode) )
        print(x/float(self.NNode))
        for istep in range(NSteps):
            for inode in self.NodeVec:
                thisamp = 0
                nbrs = inode.Neighbors
                for inbr in nbrs:                    
                    if self.NodeVec[inbr].PrevAmplitude  != 0:
                        thisamp += 1
                    
                inode.Amplitude = np.min([1,thisamp])
            self.UpdatePrev()
            x = self.SumAmp(False)
            myretarr.append( x/float(self.NNode))
            print(x/float(self.NNode))
            #print("Touch ", istep)
        return myretarr


    def MakeParity(self, Id=0):
        def IsAllDefined():
            for i in self.NodeVec:
                if i.Parity == 0:
                    return False
            return True

        self.NodeVec[0].Parity = 1
        StepChunk = 100
        while not(IsAllDefined()):
            for istep in range(StepChunk):
                for inode in self.NodeVec:
                    #import pdb; pdb.set_trace()
                    if inode.Parity != 0:
                        nbrs = [jnode for jnode in inode.Neighbors]
                        for inbr in nbrs:
                            if self.NodeVec[inbr].Parity == inode.Parity:
                                print("failed at ", inode.Id)
                                #import pdb; pdb.set_trace()
                                return
                            self.NodeVec[inbr].Parity = -inode.Parity

    def ClearParity(self, ):
        for inode in self.NodeVec:
            inode.Parity = 0



    def HeatEq(self, NSteps, iprint=0, bPrint=True):
        print("Heat")
        myretarr = [ self.NodeVec[iprint].Amplitude ]
        print( self.NodeVec[iprint].Amplitude )

        self.UpdatePrev()
        self.SumAmp(False)
        for istep in range(NSteps):
            for inode in self.NodeVec:
                thisamp = 0
                nbrs =copy(inode.Neighbors)
                for inbr in nbrs:
                    denomin = 1.0/len(self.NodeVec[inbr].Neighbors)
                    thisamp += self.NodeVec[inbr].PrevAmplitude * denomin
                inode.Amplitude = thisamp

            self.UpdatePrev()
            if True: #istep % 2 == 1:
                #print(self.NodeVec[self.NNode//2-1].Amplitude)
                if bPrint:
                    print( self.NodeVec[iprint].Amplitude )
                myretarr.append( self.NodeVec[iprint].Amplitude )
            
        return myretarr



                    

    def Wipe(self, ):
        for inode in self.NodeVec:
            inode.PrevAmplitude = 0
            inode.Amplitude = 0


                       



    def CreateNode(self,):
        newnode = node_t(self.NNode)        
        self.NodeVec.append(newnode)

        if len(self.NodeVec) != self.NNode + 1:
            print("something wrong -- go back to using NodeDict?")
            import pdb; pdb.set_trace()

        self.NNode += 1
        return newnode

    def BaseN(self, i, N=4, padlength=-1):
        ix = copy(i)
        retvec = []
        if i == 0:
            if padlength != -1:
                return [0] * padlength
            else:
                return [0]
        while ix > 0:
            thispart = ix % N
            retvec.append(thispart) # we'll reverse at the end
            ix = ix // N     
        if padlength > 0 and len(retvec) < padlength:
            padding = padlength - len(retvec)
            retvec = retvec + ([0] * padding)
        retvec.reverse()
        return retvec


    def GetAllPaths(self, x, N):
        paths = [[x]]
        if N == 1:
            return paths
        for pathlen in range(1, N+1):
            pathsNplus1 = []
            for ipath in paths:
                for inbr in self.NodeVec[ipath[-1]].Neighbors:
                    pathsNplus1.append(ipath + [inbr])
            paths = copy(pathsNplus1)
        return paths

    def Create2StepPaths(self,):
        """
        For every distinct pair of neighbor nodes of a given node whose index is x, find
        a path of length 2 (and not containing A) connecting them. If a path
        of length 2 cannot be found, try a path of length 4.
        """

        for x in range(self.NNode):
            thisnode = self.NodeVec[x]
            minpaths = {}

            for ii, i in enumerate(thisnode.Neighbors):
                
                for jj in range(ii+1, len(thisnode.Neighbors)):
                    j = thisnode.Neighbors[jj]

                    allpaths = self.GetAllPaths(i, 2)
                    
                    for ipath in allpaths:
                        if ipath[-1] == j and not(x in ipath):
                            if (i,j) in minpaths.keys():
                                minpaths[(i,j)].append(ipath)
                            else:
                                minpaths[(i,j)] = [ipath]
                    
                    #if len(allpaths) == 0:
                    if not((i,j) in minpaths):
                        allpaths = self.GetAllPaths(i, 4)
                        
                        for ipath in allpaths:
                            if ipath[-1] == j and not(x in ipath):
                                if (i,j) in minpaths.keys():
                                    minpaths[(i,j)].append(ipath)
                                else:
                                    minpaths[(i,j)] = [ipath]
                        
            thisnode.MinPaths = minpaths

    def ExtractAmplitudes(self, x, i, j):
        return x

    def AdjustAmplitudes(self, x, i, j):
        return x

    def DetermineOffsetWhorl(self, x, i, j):
        pass


def ZeroNode(ggraph):
    for inod in ggraph.NodeVec:
        inod.Parity = 0
        N = len(inod.Neighbors)
        inod.PrevAmplitude = [0 for i in range(N)]
        inod.Amplitude = [0 for i in range(N)]
        inod.ContAmplitude = [0 for i in range(N)]
        inod.AmplitudeCtl = [0 for i in range(N)]
        if ggraph.bNeedZTokens:
            inod.ZAmplitude = [0 for i in range(N)]
            #inod.ZAmplitudeIn = [0 for i in range(N)]

def SumAmp(ggraph, bAbs=False):
    sumamp = 0
    for inode in ggraph.NodeVec:
        if bAbs:
            sumamp += np.sum(np.abs(inode.Amplitude))
        else:
            sumamp += np.sum(inode.Amplitude)
    return sumamp


def Initialize(ggraph, opts):
    """ 
    Used primarily for random graphs. Note this is worthless for convergence studies because it will be 
    initialized to something different each run, but this can easily be adjusted.
    """

    if opts.loadadjacencyfile != "" or opts.loadpicklefile != "":
        ggraph = CreateGraph(opts)
        if opts.dynamics in ('pauli', 'fermion', 'fermi', 'fermi-dirac', 'fermidirac','whorl'):
            ggraph.bNeedZTokens = True   
        return ggraph


    ZeroNode(ggraph)
    sumamp = 0

    nonzeroamplist = []

    prob = opts.initprob
    if ggraph.NDim > 0:
        lngth = opts.length
    else:
        lngth = ggraph


    arngshuff = np.arange(lngth)
    rn.shuffle(arngshuff)

    #irand = -1
    for prei in range(0,lngth,2):
        #irand += 1
        # assign incoming amplitudes
        thisrand = np.random.rand()        
        if thisrand >= prob:
            continue

        #we do this in pairs, to make the symmetry more obvious (note we randomized the indices)
        # do this one
        if sumamp == 0:
            bNegate = rn.choice([True, False])
        elif sumamp > 0:
            bNegate = True
        else:
            bNegate = False

        i = arngshuff[prei]
        ich = rn.randint(0, len(ggraph.NodeVec[i].Neighbors) - 1)
        
        ggraph.NodeVec[i].PrevAmplitude[ich] = ggraph.NodeVec[i].Amplitude[ich]
        thisadd =  -1 if bNegate else 1 
        sumamp += thisadd
        ggraph.NodeVec[i].Amplitude[ich] += thisadd
        nonzeroamplist.append(i)

        # do next one
        bNegate = not(bNegate)

        i = arngshuff[prei+1]
        ich = rn.randint(0, len(ggraph.NodeVec[i].Neighbors) - 1)
        
        ggraph.NodeVec[i].PrevAmplitude[ich] = ggraph.NodeVec[i].Amplitude[ich]
        thisadd =  -1 if bNegate else 1 
        sumamp += thisadd
        ggraph.NodeVec[i].Amplitude[ich] += thisadd
        ggraph.NodeVec[i].AmplitudeCtl[ich] += thisadd
        nonzeroamplist.append(i)


    #import pdb; pdb.set_trace()
    if sumamp == 0:
        prei = 0
        i = arngshuff[prei]
        bNegate = False
        ich = rn.randint(0, len(ggraph.NodeVec[i].Neighbors) - 1)
        
        ggraph.NodeVec[i].PrevAmplitude[ich] = ggraph.NodeVec[i].Amplitude[ich]
        thisadd = 1 
        sumamp += thisadd
        ggraph.NodeVec[i].Amplitude[ich] += thisadd
        nonzeroamplist.append(i)

        # do next one
        bNegate = not(bNegate)

        i = arngshuff[prei+1]
        ich = rn.randint(0, len(ggraph.NodeVec[i].Neighbors) - 1)
        
        ggraph.NodeVec[i].PrevAmplitude[ich] = ggraph.NodeVec[i].Amplitude[ich]
        thisadd =  -1 if bNegate else 1 
        sumamp += thisadd
        ggraph.NodeVec[i].Amplitude[ich] += thisadd
        nonzeroamplist.append(i)

    if sumamp != SumAmp(ggraph):
        print("sumamp is wrong")
        import pdb; pdb.set_trace()

    if sumamp % 2 != 0:
        rn.shuffle(nonzeroamplist)
        for i in nonzeroamplist:
            if sumamp % 2 == 0:
                break
            for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
                if ggraph.NodeVec[i].Amplitude[iinbr] != 0:

                    thisadd = int(np.sign(ggraph.NodeVec[i].Amplitude[iinbr]))
                    ggraph.NodeVec[i].Amplitude[iinbr] -= thisadd
                    sumamp -= thisadd
                    break
    rn.shuffle(nonzeroamplist)

    tries = 0
    sgn = np.sign(sumamp)
    while sumamp != 0 and tries < ggraph.NNode:
        tries += 1
        for i in nonzeroamplist:
            if sumamp == 0:
                break
            for inbr in range(len(ggraph.NodeVec[i].Neighbors)):
                thisnbrsign = int(np.sign(ggraph.NodeVec[i].Amplitude[inbr]))

                if ggraph.NodeVec[i].Amplitude[inbr] != 0 and (thisnbrsign == sgn):                    
                    ggraph.NodeVec[i].Amplitude[inbr] -=  2*thisnbrsign
                    sumamp -= 2*thisnbrsign
                    if sumamp == 0:
                        break

    bPrintNonZeroInitAmplitude = False
    if bPrintNonZeroInitAmplitude:
        for i in range(ggraph.NNode):
            for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
                if ggraph.NodeVec[i].Amplitude[iinbr] != 0:
                    print("%d %d " % (i, inbr))

    thisabs = 0
    thisl2 = 0
    for i in range(ggraph.NNode):
        thisabs += np.sum(np.abs(ggraph.NodeVec[i].Amplitude))
        thisl2 += np.sum([(x * x) for x in ggraph.NodeVec[i].Amplitude])
    return ggraph

def FindNode(ggraph, coords):
    for i in range(ggraph.NNode):
        inode = ggraph.NodeVec[i]
        icoords = inode.Coords
        if tuple(coords) == tuple(icoords):
            return i 

    return -1

def FindMaxAmplitudes(ggraph, level, bSupress):
    for inode in ggraph.NodeVec:
        if np.max(inode.Amplitude) < level and -np.min(inode.Amplitude) < level:
            continue
        ampstr = ''
        for iamp in inode.Amplitude:
            if np.abs(iamp) < level and bSupress:
                ampstr = ampstr + '0.00, '
            else:
                ampstr = ampstr + '%5.5f, ' % (iamp,)
        print(inode.Id, inode.Coords, ampstr )












"""

Before and after adding an token/ztoken pair at two arcs of a configuration
showing how shifting particles (by adding a whorl) works.
                             \\
 +                    ========\\       +   
 +                    ========//       +   -            +
---  ---    ---  ---         //       ---  ---    ---  ---
reg  tok    reg  tok                  reg  tok    reg  tok
=Arc 1=     =Arc 2=                   =Arc 1=     =Arc 2=


By adding a whorl (or in this case, the inital token/antitoken pair), 
you turn the remaining regular particle on the stack (when paired/loaded with the remaining opposite-sign token) 
into part of a whorl, thereby making it vanish, so to speak (so it no longer breaches PEP). 
Also, you introduce a raw (unloaded) token on some other arc, which is effectively the same
 (in terms of what happens at the destination outgoing arc)
as having a positive particle. So you have in that sense preserved the PE whereas previously it was
violated.

"""


##### BEGIN section dealing with MODULES FOR FILLING IN (or dwindling away) a given configuratin by the addition of opposing particles.


def FillIn(Xarr):


    N = len(Xarr)
    pospart, negpart = SplitPosNeg(Xarr)

    Npos = np.sum(pospart)
    Nneg = np.sum(negpart)

    if Npos == -Nneg:
        posout = np.zeros(N).astype("int")
        return posout #, 0 #Npos

    elif Npos >= -Nneg:
        posout, leftover = FillInShell(pospart, -Nneg)
        return posout #, leftover
    else:
        negout, leftover = FillInShell(-negpart, Npos)
        return -negout #, leftover
        



def FillInShell(Xarr, Ng):

    """
    Imagine a vector of N integers, all of the same sign or zero. We want to reduce
    the sum of the vector by N in some systematic fashion. This is the workhorse function for
    dealing with either bosons or fermions. We start with an array showing both the positive breaches of our maximum
    limit per arc (which for fermions is 1 and for bosons is 1 or more) and the negative breaches.
    
    We first fill those in. Then (and let us assume it is the positive breaching that is greater in number),
    we fill in with the components of Xarr that are less than M, and thereby attempt to fill particles in so that
    there are no breaches on either side.


    We have lots of ways to do this
    but prefer something as computationally simple as possible, and will consider three alternatives; feel free
    to use.

    1) take a random permutation of 1..N and do a for loop through each index and deposit as much as you can in each
    vector before moving on. (This is the one to try first.) FillInComponentByComponent

    2) randomly choose from all the particles represented by Xarr. Since the sum of Xarr might be large, this would
    require considerable complexity to randomize/sort all the particles, so we'll skip that. FillInRandomly

    3) from  1 to Ng, choose from among Nremaining, where Nremaining is the number of components of Xarr that is nonzero.
    As a given index of Xarr is exhausted, Nremaining  is  reduced. We will try this too but still regard the amount
    of sophistication required to be suspiciously high. FillInFromBase

    4) like the previous, but fills from the top instead of the bottom.


    """

    if Ng == 0:
        return Xarr, Ng

    if np.max(Xarr) *  np.min(Xarr) < 0:
        print("ERROR in FillIn(); the first argument's components must be the same sign or zero.")
        import pdb;  pdb.set_trace()

    
    # you may also choose FillInRandomly and FillInFromBase instead
    #ThisFillInFn = FillInComponentByComponent 
    #ThisFillInFn = FillInFromPeak
    #ThisFillInFn = FillInRandomly
    ThisFillInFn = FillInFromPeak

    origNg = copy(Ng)

    sumx = np.sum(Xarr)
    if Ng >= sumx:
        return np.zeros(len(Xarr)).astype("int"), Ng - sumx

    bNegate = False
    for ix in Xarr:
        if ix < 0:
            bNegate = True
            break

    if bNegate:
        XarrNew = -Xarr
        Ng = -Ng
        FilledIn, Ngfilled = ThisFillInFn(XarrNew, Ng)
        return -FilledIn, -Ngfilled
    else:
        Xarrcp = copy(Xarr)
        return ThisFillInFn(Xarrcp, Ng)


    
    #return FillInComponentByComponent(Xarr, Ng)



def FillInComponentByComponent(Xarr, Ng):
    """
    If the amount to be removed is almost equal to the sum of what's in Xarr, then it
    would of course be easier to choose what NOT to take away, but for now, the working 
    assumption will be that Ng is small, and will be coded with that in mind.
    """

    N = len(Xarr)


    
    arng = np.arange(N)
    np.random.shuffle(arng)

    for i in arng:
        if Ng <= Xarr[i]:
            Xarr[i] -= Ng
            Ng = 0
            return Xarr, Ng
        Ng -= Xarr[i]
        Xarr[i] = 0
    
    return Xarr, Ng

def FillInRandomly(Xarr, Ng):
    """
    If the amount to be removed is almost equal to the sum of what's in Xarr, then it
    would of course be easier to choose what NOT to take away, but for now, the working 
    assumption will be that Ng is small, and will be coded with that in mind.
    """

    N = len(Xarr)

    sumx = np.sum(Xarr)
    if Ng >= sumx:
        return np.zeros(N).astype("int"), Ng - sumx

    indlist = np.random.choice(sumx, np.min([sumx, Ng]), False)
    indlist.sort()

    howmanylaidoff = 0
    howmany = 0
    for iix, ix in enumerate(Xarr):
        if ix == 0:
            continue
        howmany += ix

        while howmany > indlist[0]:
            Xarr[iix] -= 1
            howmanylaidoff += 1
            if len(indlist) == 1 or howmanylaidoff == Ng:
                break
            indlist = indlist[1:]
        if len(indlist) == 0 or howmanylaidoff == Ng:
            break

    return Xarr, Ng - howmanylaidoff




def FillInFromBase(Xarr, Ng):

    N = len(Xarr)

    sumx = np.sum(Xarr)
    if Ng >= sumx:
        return np.zeros(N).astype("int"), Ng - sumx

    if N == 0:
        return [], 0
    if N == 1:
        return [Xarr[0] - sumx], Ng


    
    while Ng > 0:
        xarrnonzero = []
        minx = copy(Ng)
        for j in range(N):
            if Xarr[j] > 0:
                xarrnonzero.append(j)
                minx =  np.min([minx, Xarr[j]])

        Nnonzero = len(xarrnonzero)
        N_thisswoop = np.min([minx*Nnonzero, Ng])
        
        
        if minx*Nnonzero > Ng:
            thisswoop = ModPoolJustBrownianAmt(Nnonzero, N_thisswoop)
        else:
            thisswoop = minx * np.ones(Nnonzero).astype("int")
        
        Ng -= N_thisswoop
        for ii,i in enumerate(xarrnonzero):
            Xarr[i] -= thisswoop[ii]

    return Xarr, Ng


def FillInFromPeak(Xarr, Ng):
    sumx = np.sum(Xarr)
    if Ng >= sumx:
        return np.zeros(N).astype("int"), Ng - sumx

    N = len(Xarr)

    if N == 0:
        return [], 0
    if N == 1:
        return [Xarr[0] - sumx], Ng

    
    sortlist = []
    for i in range(N):
        sortlist.append((-Xarr[i], i))

    sortlist.sort()
    sortlist = [x[1] for x in sortlist]


    for i in range(N):
        thisamt = Xarr[sortlist[i]]
        if i == N-1:
            thiscapacity = copy(thisamt)
        else:
            thiscapacity = Xarr[sortlist[i]] - Xarr[sortlist[i+1]]
        if thiscapacity == 0:
            continue

        columnswithmore = []
        for j in range(N):
            if Xarr[j] >= thisamt:
                columnswithmore.append(j)
        
        lencolumnswithmore = len(columnswithmore)
        thisoffset = np.min([thiscapacity * lencolumnswithmore, Ng])

        if thiscapacity * lencolumnswithmore > Ng:
            offsetarr = ModPoolJustBrownianAmt(lencolumnswithmore, thisoffset)
        else:
            offsetarr = thiscapacity * np.ones(lencolumnswithmore).astype("int")


        for ii, i in enumerate(columnswithmore):
            Xarr[i] -= offsetarr[ii]

        Ng -= thisoffset
        if Ng == 0:
            return Xarr, 0
    
    return Xarr, Ng



##### END of section dealing with MODULES FOR FILLING IN (or dwindling away) a given configuratin by the addition of opposing particles.























# BEGIN various initialization configurations




def InitializeDiracDelta(ggraph, opts, x0_coords, continuous=False):
    ZeroNode(ggraph)
    if ggraph.NDim > 0:
        i = FindNode(ggraph, x0_coords)
    else:
        i = 0
    #print(x0_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 1
    else:
        ggraph.NodeVec[i].Amplitude[0] = 1
        ggraph.NodeVec[i].AmplitudeCtl[0] = 1



def InitializeDiracDelta2(ggraph, opts, x0_coords, continuous=False):
    ZeroNode(ggraph)
    fnname = 'InitializeDiracDelta2'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 2
    else:
        ggraph.NodeVec[i].Amplitude[0] = 2
        ggraph.NodeVec[i].AmplitudeCtl[0] = 2
    #ggraph.NodeVec[i].Amplitude[2] = -1


def InitializeDiracDelta4(ggraph, opts, x0_coords, continuous=False):
    ZeroNode(ggraph)
    fnname = 'InitializeDiracDelta4'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 4
    else:
        ggraph.NodeVec[i].Amplitude[0] = 4
        ggraph.NodeVec[i].AmplitudeCtl[0] = 4
    #ggraph.NodeVec[i].Amplitude[2] = -1




def InitializeWhorl(ggraph, opts, x0_coords, continuous=False):
    InitAmp = 64 # can be ANY integer
    ZeroNode(ggraph)
    fnname = 'InitializeWhorl'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    # add a whorl at i in the 0 and 3 direction (feel free to alter all that around)
    WhorlAmp = 2
    ggraph.NodeVec[i].Amplitude[0] += WhorlAmp
    ggraph.NodeVec[i].ZAmplitude[0] += WhorlAmp
    ggraph.NodeVec[i].Amplitude[3] += WhorlAmp
    ggraph.NodeVec[i].ZAmplitude[3] += WhorlAmp





def InitializeNaiveRegEmission(ggraph, opts, x0_coords, continuous=False):
    InitAmp = 64 # can be ANY integer
    ZeroNode(ggraph)
    fnname = 'InitializeNaiveRegEmission'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    x1_coords = list(x0_coords)
    x1_coords[0] += 2
    j = FindNode(ggraph, x1_coords)


    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = InitAmp
        ggraph.NodeVec[j].ContAmplitude[1] = -InitAmp
    else:
        ggraph.NodeVec[i].Amplitude[0] = InitAmp
        ggraph.NodeVec[i].AmplitudeCtl[0] = InitAmp
        ggraph.NodeVec[j].Amplitude[1] = -InitAmp
        ggraph.NodeVec[j].AmplitudeCtl[1] = -InitAmp
    #ggraph.NodeVec[i].Amplitude[2] = -1



def InitializeNaiveRegEmissionWWhorl(ggraph, opts, x0_coords, continuous=False):
    InitAmp = 64 
    ZeroNode(ggraph)
    fnname = 'InitializeNaiveRegEmissionWWhorl'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    x1_coords = list(x0_coords)
    x1_coords[0] += 2
    j = FindNode(ggraph, x1_coords)


    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = InitAmp
        ggraph.NodeVec[j].ContAmplitude[1] = -InitAmp

    else:
        ggraph.NodeVec[i].Amplitude[0] = InitAmp
        ggraph.NodeVec[i].AmplitudeCtl[0] = InitAmp
        ggraph.NodeVec[j].Amplitude[1] = -InitAmp
        ggraph.NodeVec[j].AmplitudeCtl[1] = -InitAmp

        # add a whorl at i in the 0 and 3 direction (feel free to alter all that around)
        WhorlAmp = 2
        ggraph.NodeVec[i].Amplitude[0] += WhorlAmp
        ggraph.NodeVec[i].ZAmplitude[0] += WhorlAmp
        ggraph.NodeVec[i].Amplitude[3] += WhorlAmp
        ggraph.NodeVec[i].ZAmplitude[3] += WhorlAmp



def InitializeNaiveRegEmissionWTokens(ggraph, opts, x0_coords, continuous=False):


    InitAmp = 32 
    ZeroNode(ggraph)
    fnname = 'InitializeNaiveRegEmissionWTokens'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()  

    i = FindNode(ggraph, x0_coords)
    x1_coords = list(x0_coords)
    x1_coords[0] += 2
    j = FindNode(ggraph, x1_coords)


    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = InitAmp
        ggraph.NodeVec[j].ContAmplitude[1] = -InitAmp

    else:
        ggraph.NodeVec[i].Amplitude[0] = InitAmp
        ggraph.NodeVec[i].AmplitudeCtl[0] = InitAmp
        ggraph.NodeVec[j].Amplitude[1] = -InitAmp
        ggraph.NodeVec[j].AmplitudeCtl[1] = -InitAmp

        # add a whorl at i in the 0 and 3 direction (feel free to alter all that around)
        WhorlAmp = -InitAmp
        ggraph.NodeVec[i].Amplitude[0] += WhorlAmp
        ggraph.NodeVec[i].ZAmplitude[0] += WhorlAmp
        ggraph.NodeVec[i].Amplitude[3] += WhorlAmp
        ggraph.NodeVec[i].ZAmplitude[3] += WhorlAmp





def InitializeIncoming3pos1neg(ggraph, opts, x0_coords, continuous=False):
    fnname = 'InitializeIncoming3pos1neg'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()     
    if ggraph.TorLen < 8:
        print("ERROR: " + fnname + " is designed for to be used with lattices with a --length parameter of at least 8.")
        sys.exit()      
    ZeroNode(ggraph)
    x1_coords = (x0_coords[0], x0_coords[1])
    x2_coords = (x0_coords[0], x0_coords[1]+2)
    x3_coords = (x0_coords[0]-1, x0_coords[1]+1)
    x4_coords = (x0_coords[0]+1, x0_coords[1]+1)
    

    i =  FindNode(ggraph, x1_coords)
    i2 = FindNode(ggraph, x2_coords)
    i3 = FindNode(ggraph, x3_coords)
    i4 = FindNode(ggraph, x4_coords)
    
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[i2].ContAmplitude[3] = -1
        ggraph.NodeVec[i3].ContAmplitude[0] = 1
        ggraph.NodeVec[i4].ContAmplitude[1] = 1
    else:
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[i2].Amplitude[3] = -1
        ggraph.NodeVec[i3].Amplitude[0] = 1
        ggraph.NodeVec[i4].Amplitude[1] = 1

        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[i2].AmplitudeCtl[3] = -1
        ggraph.NodeVec[i3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[i4].AmplitudeCtl[1] = 1
    #import pdb; pdb.set_trace()


def Initialize8off(ggraph, opts, x0_coords, continuous=False):
      
    fnname = 'Initialize8off'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()     
    if ggraph.TorLen < 8:
        print("ERROR: " + fnname + " is designed for to be used with lattices with a --length parameter of at least 8.")
        sys.exit()       
    ZeroNode(ggraph)

    x0_coords = (x0_coords[0]-1, x0_coords[1]-1)
    x1_coords = (x0_coords[0], x0_coords[1])
    x2_coords = (x0_coords[0], x0_coords[1]+2)
    x3_coords = (x0_coords[0]-1, x0_coords[1]+1)
    x4_coords = (x0_coords[0]+1, x0_coords[1]+1)

    y0_coords = x2_coords
    y1_coords = x2_coords
    y2_coords = (y0_coords[0], y0_coords[1]+2)
    y3_coords = (y0_coords[0]-1, y0_coords[1]+1)
    y4_coords = (y0_coords[0]+1, y0_coords[1]+1)

    z0_coords = (x0_coords[0]-1, x0_coords[1]+1)
    z1_coords = x2_coords
    z2_coords = (y0_coords[0], y0_coords[1]+2)
    z3_coords = (y0_coords[0]-1, y0_coords[1]+1)
    z4_coords = (y0_coords[0]+1, y0_coords[1]+1)

    i =  FindNode(ggraph, x1_coords)
    i2 = FindNode(ggraph, x2_coords)
    i3 = FindNode(ggraph, x3_coords)
    i4 = FindNode(ggraph, x4_coords)

    j =  FindNode(ggraph, y1_coords)
    j2 = FindNode(ggraph, y2_coords)
    j3 = FindNode(ggraph, y3_coords)
    j4 = FindNode(ggraph, y4_coords)

    k = FindNode(ggraph, y1_coords)                        # , 11
    k2 =  FindNode(ggraph, (x2_coords[0]-1, x2_coords[1]+1)) #   13
    k3 =  FindNode(ggraph, (x2_coords[0]-2, x2_coords[1]))   # 4
    k4 =  FindNode(ggraph, y1_coords)                       # 20 # ****

    m =  FindNode(ggraph, (x2_coords[0]+1, x2_coords[1]-1)) # 27
    m2 =  FindNode(ggraph, (x2_coords[0]+1, x2_coords[1]+1)) #  29
    m3 = FindNode(ggraph, y1_coords)                        # 20 # ****
    m4 = FindNode(ggraph, (y1_coords[0]+2, y1_coords[1]))  #  36
    
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[i2].ContAmplitude[3] = -1
        ggraph.NodeVec[i3].ContAmplitude[0] = 1
        ggraph.NodeVec[i4].ContAmplitude[1] = 1

        ggraph.NodeVec[j].ContAmplitude[2] = -1
        ggraph.NodeVec[j2].ContAmplitude[3] = 1
        ggraph.NodeVec[j3].ContAmplitude[0] = 1
        ggraph.NodeVec[j4].ContAmplitude[1] = 1

        ggraph.NodeVec[k].ContAmplitude[2] = -1
        ggraph.NodeVec[k2].ContAmplitude[3] = -1
        ggraph.NodeVec[k3].ContAmplitude[0] = -1
        ggraph.NodeVec[k4].ContAmplitude[1] = -1 * -1

        ggraph.NodeVec[m].ContAmplitude[2] = 1
        ggraph.NodeVec[m2].ContAmplitude[3] = 1
        ggraph.NodeVec[m3].ContAmplitude[0] = -1
        ggraph.NodeVec[m4].ContAmplitude[1] = 1



    else:
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[i2].Amplitude[3] = -1
        ggraph.NodeVec[i3].Amplitude[0] = 1
        ggraph.NodeVec[i4].Amplitude[1] = 1

        ggraph.NodeVec[j].Amplitude[2] = -1
        ggraph.NodeVec[j2].Amplitude[3] = 1
        ggraph.NodeVec[j3].Amplitude[0] = 1
        ggraph.NodeVec[j4].Amplitude[1] = 1

        ggraph.NodeVec[k].Amplitude[2] = -1
        ggraph.NodeVec[k2].Amplitude[3] = -1
        ggraph.NodeVec[k3].Amplitude[0] = -1
        ggraph.NodeVec[k4].Amplitude[1] = +1

        ggraph.NodeVec[m].Amplitude[2] = 1
        ggraph.NodeVec[m2].Amplitude[3] = 1
        ggraph.NodeVec[m3].Amplitude[0] = -1
        ggraph.NodeVec[m4].Amplitude[1] = 1



        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[i2].AmplitudeCtl[3] = -1
        ggraph.NodeVec[i3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[i4].AmplitudeCtl[1] = 1

        ggraph.NodeVec[j].AmplitudeCtl[2] = -1
        ggraph.NodeVec[j2].AmplitudeCtl[3] = 1
        ggraph.NodeVec[j3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[j4].AmplitudeCtl[1] = 1


        ggraph.NodeVec[k].AmplitudeCtl[2] = -1
        ggraph.NodeVec[k2].AmplitudeCtl[3] = -1
        ggraph.NodeVec[k3].AmplitudeCtl[0] = -1
        ggraph.NodeVec[k4].AmplitudeCtl[1] = +1

        ggraph.NodeVec[m].AmplitudeCtl[2] = 1
        ggraph.NodeVec[m2].AmplitudeCtl[3] = 1
        ggraph.NodeVec[m3].AmplitudeCtl[0] = -1
        ggraph.NodeVec[m4].AmplitudeCtl[1] = 1


def Initialize8offA(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize8offA'
    if ggraph.NDim != 2:
        print("ERROR: " + fnname + " is designed for use only with 2-D rectangular lattice. For anything, code your own version.")
        sys.exit()        
    ZeroNode(ggraph)

    x0_coords = (x0_coords[0]-1, x0_coords[1]-1)
    x1_coords = (x0_coords[0], x0_coords[1])
    x2_coords = (x0_coords[0], x0_coords[1]+2)
    x3_coords = (x0_coords[0]-1, x0_coords[1]+1)
    x4_coords = (x0_coords[0]+1, x0_coords[1]+1)

    y0_coords = x2_coords
    y1_coords = x2_coords
    y2_coords = (y0_coords[0], y0_coords[1]+2)
    y3_coords = (y0_coords[0]-1, y0_coords[1]+1)
    y4_coords = (y0_coords[0]+1, y0_coords[1]+1)
    

    i =  FindNode(ggraph, x1_coords)
    i2 = FindNode(ggraph, x2_coords)
    i3 = FindNode(ggraph, x3_coords)
    i4 = FindNode(ggraph, x4_coords)

    j =  FindNode(ggraph, y1_coords)
    j2 = FindNode(ggraph, y2_coords)
    j3 = FindNode(ggraph, y3_coords)
    j4 = FindNode(ggraph, y4_coords)


    k = FindNode(ggraph, y1_coords)                        # , 11
    k2 =  FindNode(ggraph, (x2_coords[0]-1, x2_coords[1]+1)) #   13
    k3 =  FindNode(ggraph, (x2_coords[0]-2, x2_coords[1]))   # 4
    k4 =  FindNode(ggraph, y1_coords)                       # 20 # ****

    m =  FindNode(ggraph, (x2_coords[0]+1, x2_coords[1]-1)) # 27
    m2 =  FindNode(ggraph, (x2_coords[0]+1, x2_coords[1]+1)) #  29
    m3 = FindNode(ggraph, y1_coords)                        # 20 # ****
    m4 = FindNode(ggraph, (y1_coords[0]+2, y1_coords[1]))  #  36
      
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[i2].ContAmplitude[3] = -1
        ggraph.NodeVec[i3].ContAmplitude[0] = 1
        ggraph.NodeVec[i4].ContAmplitude[1] = 1

        ggraph.NodeVec[j].ContAmplitude[2] = -1
        ggraph.NodeVec[j2].ContAmplitude[3] = 1
        ggraph.NodeVec[j3].ContAmplitude[0] = 1
        ggraph.NodeVec[j4].ContAmplitude[1] = 1


        ggraph.NodeVec[k].ContAmplitude[2] = 1
        ggraph.NodeVec[k2].ContAmplitude[3] = 1
        ggraph.NodeVec[k3].ContAmplitude[0] = 1
        ggraph.NodeVec[k4].ContAmplitude[1] = -1 

        ggraph.NodeVec[m].ContAmplitude[2] = 1
        ggraph.NodeVec[m2].ContAmplitude[3] = 1
        ggraph.NodeVec[m3].ContAmplitude[0] = -1
        ggraph.NodeVec[m4].ContAmplitude[1] = 1



    else:
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[i2].Amplitude[3] = -1
        ggraph.NodeVec[i3].Amplitude[0] = 1
        ggraph.NodeVec[i4].Amplitude[1] = 1

        ggraph.NodeVec[j].Amplitude[2] = -1
        ggraph.NodeVec[j2].Amplitude[3] = 1
        ggraph.NodeVec[j3].Amplitude[0] = 1
        ggraph.NodeVec[j4].Amplitude[1] = 1


        ggraph.NodeVec[k].Amplitude[2] = 1
        ggraph.NodeVec[k2].Amplitude[3] = 1
        ggraph.NodeVec[k3].Amplitude[0] = 1
        ggraph.NodeVec[k4].Amplitude[1] = -1

        ggraph.NodeVec[m].Amplitude[2] = 1
        ggraph.NodeVec[m2].Amplitude[3] = 1
        ggraph.NodeVec[m3].Amplitude[0] = -1
        ggraph.NodeVec[m4].Amplitude[1] = 1



        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[i2].AmplitudeCtl[3] = -1
        ggraph.NodeVec[i3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[i4].AmplitudeCtl[1] = 1

        ggraph.NodeVec[j].AmplitudeCtl[2] = -1
        ggraph.NodeVec[j2].AmplitudeCtl[3] = 1
        ggraph.NodeVec[j3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[j4].AmplitudeCtl[1] = 1


        ggraph.NodeVec[k].AmplitudeCtl[2] = 1
        ggraph.NodeVec[k2].AmplitudeCtl[3] = 1
        ggraph.NodeVec[k3].AmplitudeCtl[0] = 1
        ggraph.NodeVec[k4].AmplitudeCtl[1] = -1

        ggraph.NodeVec[m].AmplitudeCtl[2] = 1
        ggraph.NodeVec[m2].AmplitudeCtl[3] = 1
        ggraph.NodeVec[m3].AmplitudeCtl[0] = -1
        ggraph.NodeVec[m4].AmplitudeCtl[1] = 1

    #import pdb; pdb.set_trace()

def Initialize2CollideHeadOn(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize2CollideHeadOn'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)
    x1_coords = (x0_coords[0], x0_coords[1])
    x2_coords = (x0_coords[0], x0_coords[1]+2)
    #x3_coords = (x0_coords[0]-1, x0_coords[1]+1)
    #x4_coords = (x0_coords[0]+1, x0_coords[1]+1)
    

    i =  FindNode(ggraph, x1_coords)
    i2 = FindNode(ggraph, x2_coords)
    #i3 = FindNode(ggraph, x3_coords)
    #i4 = FindNode(ggraph, x4_coords)
    
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[i2].ContAmplitude[3] = -1
        #ggraph.NodeVec[i3].ContAmplitude[0] = 1
        #ggraph.NodeVec[i4].ContAmplitude[1] = 1
    else:
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[i2].Amplitude[3] = -1
        #ggraph.NodeVec[i3].Amplitude[0] = 1
        #ggraph.NodeVec[i4].Amplitude[1] = 1

        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[i2].AmplitudeCtl[3] = -1
        #ggraph.NodeVec[i3].AmplitudeCtl[0] = 1
        #ggraph.NodeVec[i4].AmplitudeCtl[1] = 1


def Initialize2CollideHeadOnSameSign(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize2CollideHeadOn'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)
    x1_coords = (x0_coords[0], x0_coords[1])
    x2_coords = (x0_coords[0], x0_coords[1]+2)
    #x3_coords = (x0_coords[0]-1, x0_coords[1]+1)
    #x4_coords = (x0_coords[0]+1, x0_coords[1]+1)
    

    i =  FindNode(ggraph, x1_coords)
    i2 = FindNode(ggraph, x2_coords)
    #i3 = FindNode(ggraph, x3_coords)
    #i4 = FindNode(ggraph, x4_coords)
    
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[i2].ContAmplitude[3] = 1
        #ggraph.NodeVec[i3].ContAmplitude[0] = 1
        #ggraph.NodeVec[i4].ContAmplitude[1] = 1
    else:
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[i2].Amplitude[3] = 1
        #ggraph.NodeVec[i3].Amplitude[0] = 1
        #ggraph.NodeVec[i4].Amplitude[1] = 1

        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[i2].AmplitudeCtl[3] = 1
        #ggraph.NodeVec[i3].AmplitudeCtl[0] = 1
        #ggraph.NodeVec[i4].AmplitudeCtl[1] = 1



def Initialize1whorlA(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize1whorlA'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 0
    else:
        
        ggraph.NodeVec[i].AmplitudeCtl[0] = 0


        ggraph.NodeVec[i].Amplitude[1] = 0

        # now add a loaded token/antitoken pair
        #import pdb; pdb.set_trace()
        ggraph.NodeVec[i].Amplitude[1] += 1
        ggraph.NodeVec[i].ZAmplitude[1] += 1
        ggraph.NodeVec[i].Amplitude[0] += -1
        ggraph.NodeVec[i].ZAmplitude[0] += -1


    #ggraph.NodeVec[i].Amplitude[2] = -1


def Initialize1shift(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize1shift'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)
    #print(x0_coords)
    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 1
    else:
        
        ggraph.NodeVec[i].AmplitudeCtl[0] = 1


        ggraph.NodeVec[i].Amplitude[0] = 1

        # now add a loaded token/antitoken pair
        ggraph.NodeVec[i].Amplitude[1] += -1
        ggraph.NodeVec[i].ZAmplitude[1] += 1
        ggraph.NodeVec[i].Amplitude[0] += -1
        ggraph.NodeVec[i].ZAmplitude[0] += -1


    #ggraph.NodeVec[i].Amplitude[2] = -1



def Initialize2offC(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize2offC'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)
    x1_coords = list(x0_coords)
    x1_coords[0] += 2
    j = FindNode(ggraph, x1_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = -1
        ggraph.NodeVec[j].ContAmplitude[1] = 1
    else:
        ggraph.NodeVec[i].Amplitude[0] = -1
        ggraph.NodeVec[j].Amplitude[1] = 1

        ggraph.NodeVec[i].AmplitudeCtl[0] = -1
        ggraph.NodeVec[j].AmplitudeCtl[1] = 1

    #ggraph.NodeVec[i].Amplitude[2] = -1



def Initialize2offD(ggraph, opts, x0_coords, continuous=False):
    fnname = 'Initialize2offD'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)
    x1_coords = list(x0_coords)
    x1_coords[0] += 2
    j = FindNode(ggraph, x1_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 1
        ggraph.NodeVec[j].ContAmplitude[1] = 1
    else:
        ggraph.NodeVec[i].Amplitude[0] = 1
        ggraph.NodeVec[j].Amplitude[1] = 1

        ggraph.NodeVec[i].AmplitudeCtl[0] = 1
        ggraph.NodeVec[j].AmplitudeCtl[1] = 1

    #ggraph.NodeVec[i].Amplitude[2] = -1



def Initialize1cluster(ggraph, opts, x0_coords):
    fnname = 'Initialize1cluster'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit()        
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)
    listx0 = list(x0_coords)

    x_upx = copy(listx0)
    x_upx[0] += 2
    i_upx = FindNode(ggraph, x_upx)

    x_upxy = copy(listx0)
    x_upxy[0] += 1
    x_upxy[1] += 1
    i_upxy = FindNode(ggraph, x_upxy)

    x_upxdny = copy(listx0)
    x_upxdny[0] += 1
    x_upxdny[1] -= 1
    i_upxdny = FindNode(ggraph, x_upxdny)


    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = -1
        ggraph.NodeVec[i_upx].ContAmplitude[1] = 1

        ggraph.NodeVec[i_upxy].ContAmplitude[3] = 1
        ggraph.NodeVec[i_upxdny].ContAmplitude[2] = 1
    else:

        ggraph.NodeVec[i].Amplitude[0] = -1
        ggraph.NodeVec[i_upx].Amplitude[1] = 1
        ggraph.NodeVec[i_upxy].Amplitude[3] = 1
        ggraph.NodeVec[i_upxdny].Amplitude[2] = 1

        ggraph.NodeVec[i].AmplitudeCtl[0] = -1
        ggraph.NodeVec[i_upx].AmplitudeCtl[1] = 1
        ggraph.NodeVec[i_upxy].AmplitudeCtl[3] = 1
        ggraph.NodeVec[i_upxdny].AmplitudeCtl[2] = 1





def Initialize1shift(ggraph, opts, x0_coords, continuous=False):
    """
    See above caution before trying to compare this to a
    function that has no tokens (or, say a ModPool realizaton)
    """    


    fnname = 'Initialize1shift'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 

    if not(ggraph.bNeedZTokens):
        print("ERROR: " + fnname + " is designed for use only with dynamical models that allow for Z-tokens.")
        sys.exit()         

    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] += 1
        #ggraph.NodeVec[i].ContAmplitude[2] += -1
    else:
        ggraph.NodeVec[i].Amplitude[0] += 1
        ggraph.NodeVec[i].AmplitudeCtl[0] += 1
        #ggraph.NodeVec[i].AmplitudeCtl[2] += -1

        # now overlay the whorl
        ggraph.NodeVec[i].Amplitude[0] += -1
        ggraph.NodeVec[i].ZAmplitude[0] += -1

        ggraph.NodeVec[i].Amplitude[2] += 1
        ggraph.NodeVec[i].ZAmplitude[2] += 1




def Initialize2shift(ggraph, opts, x0_coords, continuous=False):
    """
    See above caution before trying to compare this to a
    function that has no tokens (or, say a ModPool realizaton)
    """    


    fnname = 'Initialize2shift'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 

    if not(ggraph.bNeedZTokens):
        print("ERROR: " + fnname + " is designed for use only with dynamical models that allow for Z-tokens.")
        sys.exit()   

    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] += 1
        ggraph.NodeVec[i].ContAmplitude[2] += -1
    else:
        ggraph.NodeVec[i].Amplitude[0] += 1
        ggraph.NodeVec[i].Amplitude[2] += -1
        ggraph.NodeVec[i].AmplitudeCtl[0] += 1
        ggraph.NodeVec[i].AmplitudeCtl[2] += -1

        # now overlay the whorl
        ggraph.NodeVec[i].Amplitude[0] += -1
        ggraph.NodeVec[i].ZAmplitude[0] += -1

        ggraph.NodeVec[i].Amplitude[2] += 1
        ggraph.NodeVec[i].ZAmplitude[2] += 1





def Initialize1Zwhorl(ggraph, opts, x0_coords, continuous=False):
    

    fnname = 'Initialize1Zwhorl'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 
    if not(ggraph.bNeedZTokens):
        print("ERROR: " + fnname + " is designed for use only with dynamical models that allow for Z-tokens.")
        sys.exit()   

    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = [i + 1 for i in x0_coords]
    j = FindNode(ggraph, x1_coords)

    if continuous:
        pass # ggraph.NodeVec[i].ContAmplitude[0] = 1
    else:
        #ggraph.NodeVec[i].Amplitude[0] = 1
        #ggraph.NodeVec[i].Amplitude[0] = -1
        #ggraph.NodeVec[i].Amplitude[2] = 1


        ggraph.NodeVec[i].ZAmplitude[0] = -1
        ggraph.NodeVec[i].ZAmplitude[2] = 1
        ggraph.NodeVec[j].ZAmplitude[3] = 1
        ggraph.NodeVec[j].ZAmplitude[1] = -1
          
        #ggraph.NodeVec[i].AmplitudeCtl[0] = 1
    #ggraph.NodeVec[i].Amplitude[2] = -1


def Initialize2off(ggraph, opts, x0_coords, continuous=False):
    
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = [i + 1 for i in x0_coords]
    j = FindNode(ggraph, x1_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = -1
        #ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[j].ContAmplitude[3] = 1
        #ggraph.NodeVec[j].ContAmplitude[1] = -1

    else:


        ggraph.NodeVec[i].Amplitude[0] = -1
        #ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[j].Amplitude[3] = 1
        #ggraph.NodeVec[j].Amplitude[1] = -1

        ggraph.NodeVec[i].AmplitudeCtl[0] = -1
        ggraph.NodeVec[j].AmplitudeCtl[3] = 1
                  
        #ggraph.NodeVec[i].AmplitudeCtl[0] = 1
    #ggraph.NodeVec[i].Amplitude[2] = -1



def Initialize2CollideAngle(ggraph, opts, x0_coords, continuous=False):


    fnname = 'Initialize2CollideAngle'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 


    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = list(x0_coords)
    x1_coords[0] += 1
    x1_coords[1] += 1
    j = FindNode(ggraph, x1_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = -1
        #ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[j].ContAmplitude[3] = 1
        #ggraph.NodeVec[j].ContAmplitude[1] = -1

    else:


        ggraph.NodeVec[i].Amplitude[0] = -1
        #ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[j].Amplitude[3] = 1
        #ggraph.NodeVec[j].Amplitude[1] = -1

        ggraph.NodeVec[i].AmplitudeCtl[0] = -1
        ggraph.NodeVec[j].AmplitudeCtl[3] = 1
                  
        #ggraph.NodeVec[i].AmplitudeCtl[0] = 1
    #ggraph.NodeVec[i].Amplitude[2] = -1



def Initialize2CollideAngleSameSign(ggraph, opts, x0_coords, continuous=False):


    fnname = 'Initialize2CollideAngleSameSign'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 


    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = list(x0_coords)
    x1_coords[0] += 1
    x1_coords[1] += 1
    j = FindNode(ggraph, x1_coords)

    if continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 1
        #ggraph.NodeVec[i].ContAmplitude[2] = 1
        ggraph.NodeVec[j].ContAmplitude[3] = 1
        #ggraph.NodeVec[j].ContAmplitude[1] = -1

    else:


        ggraph.NodeVec[i].Amplitude[0] = 1
        #ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[j].Amplitude[3] = 1
        #ggraph.NodeVec[j].Amplitude[1] = -1

        ggraph.NodeVec[i].AmplitudeCtl[0] = 1
        ggraph.NodeVec[j].AmplitudeCtl[3] = 1
                  
        #ggraph.NodeVec[i].AmplitudeCtl[0] = 1
    #ggraph.NodeVec[i].Amplitude[2] = -1






def Initialize1plaquette(ggraph, opts, x0_coords, continuous=False):
    
    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = [i + 1 for i in x0_coords]
    j = FindNode(ggraph, x1_coords)

    if False: #continuous:
        ggraph.NodeVec[i].ContAmplitude[0] = 1
    else:
        ggraph.NodeVec[i].Amplitude[0] = -1
        ggraph.NodeVec[i].Amplitude[2] = 1
        ggraph.NodeVec[j].Amplitude[3] = 1
        ggraph.NodeVec[j].Amplitude[1] = -1

        ggraph.NodeVec[i].AmplitudeCtl[0] = -1
        ggraph.NodeVec[i].AmplitudeCtl[2] = 1
        ggraph.NodeVec[j].AmplitudeCtl[3] = 1
        ggraph.NodeVec[j].AmplitudeCtl[1] = -1





def Initialize1plaquetteZ(ggraph, opts, x0_coords, continuous=False):



    fnname = 'Initialize1plaquetteZ'
    if ggraph.NDim == 0:
        print("ERROR: " + fnname + " is designed for use only with rectangular lattice. If you're using something else, you'll need to code your own initialization routine.")
        sys.exit() 




    ZeroNode(ggraph)

    i = FindNode(ggraph, x0_coords)

    x1_coords = [i + 1 for i in x0_coords]
    j = FindNode(ggraph, x1_coords)

    if continuous:
        #ggraph.NodeVec[i].ContAmplitude[0] = 1

        #ggraph.NodeVec[i].AmplitudeCtl[0] = +1
        #ggraph.NodeVec[i].AmplitudeCtl[2] = -1
        #ggraph.NodeVec[j].AmplitudeCtl[3] = -1
        #ggraph.NodeVec[j].AmplitudeCtl[1] = +1
        pass

    else:
        ggraph.NodeVec[i].ZAmplitude[0] = -1
        ggraph.NodeVec[i].ZAmplitude[2] = 1
        ggraph.NodeVec[j].ZAmplitude[3] = 1
        ggraph.NodeVec[j].ZAmplitude[1] = -1

        #ggraph.NodeVec[i].AmplitudeCtl[0] = +1
        #ggraph.NodeVec[i].AmplitudeCtl[2] = -1
        #ggraph.NodeVec[j].AmplitudeCtl[3] = -1
        #ggraph.NodeVec[j].AmplitudeCtl[1] = +1









# END of section dealing with various initialization configurations
















def SaveGraph(ggraph, dumppklname):
    with open(dumppklname, 'wb') as file:
        graphobj = pickle.dump([ggraph], file) # we save it is a 1-element list to make the loading smooth regardless of whether the graph was saved with sys.argv or not



def SaveGraphWOpt(ggraph, dumppklname):
    with open(dumppklname, 'wb') as file:
        graphobj = pickle.dump([ggraph, ' '.join(sys.argv)], file)



def SaveAdjacencyMatrix(ggraph, csvfilename):
    """ 
    Note the saved file is an actual NxN matrix (and therefore does not save any amplitude info)
    """
    
    AdjMat = np.zeros(ggraph.NNode, ggraph.NNode)
    for i in range(ggraph.NNode):
        for inbr in ggraph.NodeVec[i].Neighbors:
            AdjMat[i, inbr] = 1
            AdjMat[inbr, i] = 1
    
    np.savetxt(csvfilename, AdjMat, fmt='%d', delimiter=',', newline='\n', header='', footer='')


def SaveAdjacencyList(ggraph, csvfilename):
    """ 
    Note the saved file, unlike what is created in SaveAdjacencyMatrix() is a list that can be read in
    to create both a graph, and an initial state. 
    """
    
    AdjMat = np.zeros(ggraph.NNode, ggraph.NNode)
    for i in range(ggraph.NNode):
        thisline = ""
        theselines = []
        for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
            precomma = ""
            if len(thisline) > 0:
                precomma = ","
            thisline = thisline + precomma + ("%d;%d;%d" % (ggraph.NodeVec[i].Neighbors[iinbr], ggraph.NodeVec[i].Amplitude[iinbr], ggraph.NodeVec[i].ZAmplitude[iinbr]))
        theselines.append(thisline)
    
    f = open(csvfilename, 'w')
    f.writelines(theselines)
    f.close()

def PrintGraphByNode(ggraph, bCont=False):
    for i in range(ggraph.NNode):
        print(i, "amp", ggraph.NodeVec[i].Amplitude, "z", ggraph.NodeVec[i].ZAmplitude, "ctl", ggraph.NodeVec[i].AmplitudeCtl, "cont", ggraph.NodeVec[i].ContAmplitude)
    

def CreateGraphFromAdjacencyList(csvfilename):
    """ 
    Assumes any amplitude information is delimited 
    by periods, or, where absent, the relevant
    particle and z-token counts are zero.
    """
    ggraph = graph_t()
    ggraph.NDim = 0

    f = open(csvfilename, 'r')
    theselines = f.readlines()
    f.close()
    for iline in theselines:
        nodechunks = iline.split(',')
        thisnode = ggraph.CreateNode()
        for ichunk in nodechunks:
            thislist = [0, 0, 0]
            thischunksplit = ichunk.split('.')
            thischunksplitnumeric = [int(j) for j in thischunksplit]
            for jj, j in enumerate(thischunksplitnumeric):
                thislist[jj] = j
            
            thisnode.Neighbors.append(thislist[0])
            thisnode.Amplitude.append(thislist[1])
            thisnode.AmplitudeCtl.append(thislist[1])
            thisnode.ContAmplitude.append(thislist[1])
            thisnode.ZAmplitude.append(thislist[2])
        thisnode.Scratch = [0 for i in nodechunks]
        thisnode.ZScratch = [0 for i in nodechunks]
        thisnode.ScratchCtl = [0 for i in nodechunks]
        thisnode.ContScratch = [0 for i in nodechunks]
        thisnode.PrevAmplitude = [0 for i in nodechunks]


    #import pdb; pdb.set_trace()
    return ggraph



def CreateGraph(opts):

    loadpklfilename = opts.loadpicklefile
    adjacencyfilename = opts.loadadjacencyfile

    if loadpklfilename != "" and adjacencyfilename != "":
        print("ERROR: --loadfrompkl and --loadfromadjacencyfile cannot be invoked simultaneously; pick the correct one and rerun")
        sys.exit()
    if loadpklfilename != "":
        with open(loadpklfilename, 'rb') as file:
            contents = pickle.load(file)
            graphobj = contents[0]
            graphobj.GetFromNodeNbr()
            return graphobj
    if adjacencyfilename != "":
        graphobj = CreateGraphFromAdjacencyList(adjacencyfilename)
        #import pdb; pdb.set_trace()
        graphobj.GetFromNodeNbr()
        return graphobj


    ggraph = graph_t()


    D = opts.dimension
    lngth = opts.length
    prob = opts.prob 
    # option A -- D-dimensional array
    if D > 0:
        ggraph.NDim = D
        ggraph.TorLen = lngth
        if lngth <= 0:
            print("If --dimension is > 0, then length (i.e. the number of nodes along each axis of cube) must be > 0; confusing instructions")
            return None
        if prob > 0:
            print("If --dimension is > 0, then --prob option must not be specified; confusing instructions")
            return None
        if opts.minmaxnbr != "0,0":
            print("If --dimension is > 0, then --minmax option must not be specified; confusing instructions")
            return None
        NNode = lngth ** D
        maxnodes = 500 * 1_000
        if NNode > maxnodes:
            print("The routine has a hard-coded maximum node number of 500,000 whereas length**D is more than that; change the limit, or change inputs; confusing instructions")
            return None
        
        itocoord = {}
        coordtoi = {}
        for i in range(NNode):
            ivec = ggraph.BaseN(i, lngth, D)
            itocoord[i] = tuple(ivec)
            coordtoi[tuple(ivec)] = i
            ggraph.CreateNode()
            ggraph.NodeVec[ggraph.NNode-1].Coords = ivec

        
        for i in range(NNode):
            thisnode = ggraph.NodeVec[i]
            ivecc = itocoord[i] 
            nbrvec = []
            
            for idir in range(D):
                upvec = list(ivecc)
                dnvec = copy(upvec)
                upvec[idir] = (ivecc[idir] + 1) % lngth
                dnvec[idir] = ivecc[idir] - 1 if ivecc[idir] > 0 else (lngth - 1)
                nbrvec.append(tuple(upvec))
                nbrvec.append(tuple(dnvec))
            
            for itup in nbrvec:
                nbrnode = coordtoi[itup]
                thisnode.Neighbors.append(nbrnode)
            #thisnode.Neighbors.sort()
        

        if opts.diagonalize:          
            DiagonalizeRectangularGraphOnce(ggraph)

        ggraph.GetFromNodeNbr()

        return ggraph



    # option B (deprecated) -- adjacency matrix_ij has a probability of being 1; MINIMUM NUMBER OF NBRS IS 3

    adjmat = np.zeros((lngth, lngth))



    # option B2 -- probability

    if opts.minmaxnbr == '' and opts.dimension == 0 and opts.loadadjacencyfile == '':
        print("No --minmaxnbr arg, will assume the min is 3 and max is equal to --length arg)")
        imin, imax = (3,lngth)
    else:
        istr,jstr = opts.minmaxnbr.split(',')
        imin = int(istr)
        imax = int(jstr)

    #if imin < 3:
    #    print("Overriding --minmaxnbr input and resetting min number of neighbors to be 3.")
    #    imin = 3
    if imax <= imin:
        imax = lngth
        imin = 3 
        print("minnbrs will arbitrarily be set to 3")

    for i in range(lngth):
        ggraph.CreateNode()




    arngshuff = np.arange(lngth)
    rn.shuffle(arngshuff)

    
    inbrs = {}  
    for prei in range(lngth):
        i = arngshuff[prei]



        if not(i in inbrs):
            inbrs[i] = []
        elif len(inbrs[i]) >= imax:
            continue

        for prej in range(i+1, lngth):
            j = arngshuff[prej]

            if not(j in inbrs):
                inbrs[j] = []
            elif len(inbrs[j]) >= imax:
                continue

            if adjmat[i, j] == 1:
                continue

            thisrand = np.random.rand()
            
            if thisrand < prob:
                adjmat[i, j] = 1
                adjmat[j, i] = 1
                inbrs[i].append(j)
                inbrs[j].append(i)
                


        
        # len(ggraph.NodeVec[0].Neighbors)
        if len(inbrs[i]) < imin:

            itry = 0
            maxtries = 500
            while itry < maxtries:
                itry += 1
                for j in arngshuff:
                    if j == i:
                        continue
                    
                    if not(j in inbrs):
                        inbrs[j] = []
                    elif len(inbrs[j]) >= imax:
                        continue

                    inbrs[i].append(j)
                    inbrs[j].append(i)                
                    
                    if len(inbrs[i]) >= imin:
                        itry = maxtries
                        break

            
        if len(inbrs[i]) < imin or len(inbrs[i]) > imax:
            print("Failed trying to construct a graph where everything has between", str(imin), "and", str(imax), "items. Try with some other pair.")
            return None


    for i in range(lngth):
        nbrlist = sorted(inbrs[i])            
        ggraph.NodeVec[i].Neighbors = nbrlist
        
    ggraph.GetFromNodeNbr()



    return ggraph                
                        
        

     










    # option C -- adjacency matrix_ij has m neighbors where m is between two inuts

    istr,jstr = opts.minmaxnbr.split(',')
    imin = int(istr)
    imax = int(jstr)
    if imax <= imin:
        print("ERROR: 2-tuple in --minmaxnbr needs to be of two integers with 2nd larger than 1st.")
        return None

    if imax > 0:
        if D <= 0:
            print("If --prob is > 0, then --dimension option must not be invoked; confusing instructions")
            return None
        inbrs = {}   

        for i in range(lngth):
            if not(i in inbrs):
                inbrs[i] = []
            elif len(inbrs[i]) >= imin:
                continue

            arngshuff = np.arange(0, lngth)
            arngshuff.shuffle()

            for j in arngshuff:
                if j == i:
                    continue
                
                if not(j in inbrs):
                    inbrs[j] = []
                elif len(inbrs[j]) >= imin:
                    continue
                

                inbrs[i].append(j)
                inbrs[j].append(i)

                if len(inbrs[i]) >= imin:
                    break
            
            if len(inbrs[i]) < imin:
                for j in arngshuff:
                    if j == i:
                        continue
                    
                    if not(j in inbrs):
                        inbrs[j] = []
                    elif len(inbrs[j]) >= imax:
                        continue
                    

                    inbrs[i].append(j)
                    inbrs[j].append(i)                

                    if len(inbrs[i]) >= imin:
                        break
            
            if len(inbrs[i]) < imin or len(inbrs[i]) > imax:
                print("Failed trying to construct a graph where everything has between", str(imin), "and", str(imax), "items. Try with some other pair.")
                return None


        for i in range(lngth):
            nbrlist = sorted(inbrs[i])
            ggraph.NodeVec[i].Neighbors = nbrlist
        ggraph.GetFromNodeNbr()
        return ggraph                
                
        
def GetNodeFromCoords(ggraph, coord):
    """
    works only for rectangular array graphs
    """
    if ggraph.NDim == 0:
        print("ERROR: GetNodeFromCoords() only works for rectangular array graphs.")
    
    tupcoord = tuple(coord)
    for iinode, inod in enumerate(ggraph.NodeVec):
        if tuple(inod.Coords) == tupcoord:
            return iinode
    return -1




def FloatingPtWave(inlist):
    # treat the transfer "wave" matrix -I + 2*H (where H is the idempotent heat matrix)
    # as a probability matrix, so that for every input particle,  each output arc gas a cgabce
    # of getting two output particle of same sign with probability 1/N each (not including the
    # determinstic Huygens particle), so  the contribution has a sum that ranges from -1 to
    # 2*N-1 (i.e. -1, for the huygens particle plus the brownian component which ranges from 0 to 2*N)

    N = len(inlist)
    outlistcont = -np.array(inlist).astype("float") + np.sum(inlist) * 2.0/N 
    return outlistcont #, err

      

def BernouilliVersion(inlist):
    # treat the transfer "wave" matrix -I + 2*H (where H is the idempotent heat matrix)
    # as a probability matrix, so that for every input particle,  each output arc gas a cgabce
    # of getting two output particle of same sign with probability 1/N each (not including the
    # determinstic Huygens particle), so  the contribution has a sum that ranges from -1 to
    # 2*N-1 (i.e. -1, for the huygens particle plus the brownian component which ranges from 0 to 2*N)
    outlist = np.array(inlist).astype("int")
    N = len(inlist)

    #outlistcont = -np.array(inlist).astype("float") + np.sum(inlist) * 2.0/N

    var = 0

    pospool = np.sum([ np.max([i, 0]) for i in inlist])
    negpool = np.sum([ np.min([i, 0]) for i in inlist])
    pospool *= 2
    negpool *= 2


    oneoverN = 1.0/N
    incNabs = 0

    #import pdb; pdb.set_trace()

    posdeviate = np.random.binomial(n=pospool, p = oneoverN, size=N) 
    negdeviate = -np.random.binomial(n=-negpool, p = oneoverN, size=N)

    outlist = posdeviate + negdeviate - inlist

    #if np.sum(outlist) != np.sum(inlist):
    #    import pdb; pdb.set_trace()
    #err = outlist - outlistcont
    #var += np.sum(err * err)
    #incNabs += np.sum(np.abs(outlist)) - np.sum(np.abs(inlist))
    
    return outlist #, err

        
    
# for alpha = 0, this generates pure B-H; for alpha = 1, we get "pure" modpool;
# i.e. this does the mixed case where alpha can be from 0 to 1
def ModPool(inlist, alpha=1):

    global probtracker
    global CUTOFF
    global CUTOFFVOLUME
    global FAC
    """
    WARNING: Since I'm not using the "mixed-case" scenario where alpha is between 0 and 1
    (and therefore not purely Brownian-Huygens with a particle growth rate ~ T and not
    purely ModPool with a particle growth rate ~ sqrt(T)), it has not been maintained, especially
    since the alpha=0 case isn't compatible with PEP (at least not in the way it has been implemented here)
    so that I'll just stick with alpha=1/Modpool dynamics.
    """

    brownian_pool = 2 * np.sum(inlist)


    # first, the Huygens portion 
    N_arc = len(inlist)

    huygportion = copy(inlist)
    outlist = copy(inlist)
    for i in range(len(inlist)):
        huygportion[i] = -inlist[i]



    if True:

        if False and alpha != 0.0 and alpha != 1.0: # the general "mixed case", handles every case of alpha,
                                            # but we'll go ahead and do the pure B-H and pure "modpooling" separately

            pospool = np.sum([ np.max([x,0]) for x in inlist]) # np.max([incoming_along_xneg,0]) + np.max([incoming_along_yneg,0]) + np.max([incoming_along_xpos,0]) + np.max([incoming_along_ypos,0])
            negpool = np.sum([ np.min([x,0]) for x in inlist]) # np.min([incoming_along_xneg,0]) + np.min([incoming_along_yneg,0]) + np.min([incoming_along_xpos,0]) + np.min([incoming_along_ypos,0])

            pospool *= 2
            negpool *= 2

            pospool = int(np.floor((1.0-alpha) * pospool))
            negpool = int(np.floor((1.0-alpha) * negpool))

            brownian_pool -= pospool + negpool

            abspool = np.abs(brownian_pool)
            sgnpool = np.sign(brownian_pool)
            modpool = abspool % N_arc
            divpool = (abspool // N_arc) 

            # ad any modulo remainder bits to either the positive-sign B-H output, or the negative-sign B-H output
            if sgnpool > 0:
                pospool += modpool
            elif sgnpool < 0:
                negpool -= modpool

            posdeviate = np.random.multinomial(pospool, [1/float(N_arc)]*N_arc)
            negdeviate = np.random.multinomial(-negpool, [1/float(N_arc)]*N_arc) 

            


            divpool *= sgnpool

            for i in range(len(inlist)):
                outdeviatemod = (huygportion[i] + posdeviate[i] - negdeviate[i])  + divpool

        
        elif alpha == 0: # the "pure" Brownian-Huygens case, for which sum(abs(grid)) grows like t

            # note that for this case we did not need the above calculation of brownianpool, since that is unused here

            pospool = np.sum([ np.max([x,0]) for x in inlist]) # np.max([incoming_along_xneg,0]) + np.max([incoming_along_yneg,0]) + np.max([incoming_along_xpos,0]) + np.max([incoming_along_ypos,0])
            negpool = np.sum([ np.min([x,0]) for x in inlist]) # np.min([incoming_along_xneg,0]) + np.min([incoming_along_yneg,0]) + np.min([incoming_along_xpos,0]) + np.min([incoming_along_ypos,0])

            pospool *= 2
            negpool *= 2

            posdeviate = np.random.multinomial(pospool, [1/float(N_arc)]*N_arc)
            negdeviate = np.random.multinomial(-negpool, [1/float(N_arc)]*N_arc) 
            for i in range(len(inlist)):
                outlist[i] = huygportion[i] + posdeviate[i] - negdeviate[i]

        
        else: # alpha == 1 (the "modpool" case for which sum(abs(grid)) grows like sqrt(t) )

            # in this case, we don't need a separate pospool and negpool, we just mod out the brownian (i.e. sum of neg and pos) pool

            abspool = np.abs(brownian_pool)
            sgnpool = np.sign(brownian_pool)
            modpool = abspool % N_arc
            divpool = (abspool // N_arc) # this (times sgnpool) is a scalar value that gets added to every output arc


            # if modpool is > N_arc//2, then we should change things so we only have to generate N_arc - modpool random variables.
            NegateModpoolFactor = 1
            if modpool > N_arc // 2:
                NegateModpoolFactor = -1
                modpool = N_arc - modpool
                divpool += 1

            # this is the randomized "leftover" part
            #outdeviatemod = sgnpool * np.random.multinomial(modpool, [1/float(N_arc)]*N_arc)

            # this will work better for fermions in that the modpool portion will be distributed to different arcs
            # i.e. they will be spread out and never lump up
            outdeviatemod = np.zeros((N_arc,)).astype("int")
            modpoolchoices = np.random.choice(N_arc, modpool, False, None)
            for i in modpoolchoices:
                outdeviatemod[i] += sgnpool * NegateModpoolFactor

            divpool *= sgnpool

            for i in range(len(inlist)):
                outlist[i] = huygportion[i] + outdeviatemod[i] + divpool

    # Naive regularization section
    
    if CUTOFF != 0.0:
        
        origoutlist = copy(outlist)
        maxoutgoing = np.max(outlist)
        minoutgoing = np.min(outlist)


        if maxoutgoing <= CUTOFF and -minoutgoing <= CUTOFF:
            pass

        elif maxoutgoing - CUTOFF >= (-minoutgoing - CUTOFF):
            whichexceed = []
            sumexcess = 0
            whichhascapacity = []
            sumcapacity = 0
            for i in range(N_arc):
                excess = np.max([0, outlist[i] - CUTOFF])

                
                if CUTOFFVOLUME != 0:
                    excess = np.min([CUTOFFVOLUME, excess])
                if excess > 0:
                    whichexceed.append((excess + np.random.normal()/100.0, i)) #
                    sumexcess += excess

                capacity = np.min([0, outlist[i]])
                if capacity < 0:
                    whichhascapacity.append((capacity + np.random.normal()/100.0, i))
                    sumcapacity += capacity

            whichexceed.sort()
            whichexceed.reverse() 
            whichhascapacity.sort()

            # let's start with the simplest choice (more or less) of just annihilating the biggest excsss with the biggest capacity
            if rn.random() < FAC:
                if sumcapacity < 0 and sumexcess > 0:
                    if int(np.round(whichexceed[0][0])) >= -int(np.round(whichhascapacity[0][0])):
                        outlist[ whichexceed[0][1] ] += int(np.round(whichhascapacity[0][0]))
                        outlist[ whichhascapacity[0][1] ] -= int(np.round(whichhascapacity[0][0]))
                    else:
                        outlist[ whichexceed[0][1] ]  -= int(np.round(whichexceed[0][0]))
                        outlist[ whichhascapacity[0][1] ] += int(np.round(whichexceed[0][0]))
        else:
            
            whichexceed = []
            sumexcess = 0
            whichhascapacity = []
            sumcapacity = 0

            # NOTE WE ADD FUDGE FACTORS TO THE EXCESS; if we don't do this then when we sort, we'll always have lower indices appear first, and we want to randomize those
            for i in range(N_arc):
                excess = np.max([0, -outlist[i] - CUTOFF])
                if CUTOFFVOLUME != 0:
                    excess = np.min([excess, CUTOFFVOLUME])
                if excess > 0:
                    whichexceed.append((excess + np.random.normal()/100., i))
                    sumexcess += excess

                capacity = np.max([0, outlist[i]])
                if capacity > 0:
                    whichhascapacity.append((capacity + np.random.normal()/100.0, i))
                    sumcapacity += capacity

            whichexceed.sort()
            whichexceed.reverse() 
            whichhascapacity.sort()

            # let's start with the simplest choice (more or less) of just annihilating the biggest excsss with the biggest capacity
            if rn.random() < FAC:                    
                if sumcapacity > 0 and sumexcess > 0:
                    if int(np.round(whichexceed[0][0])) >= int(np.round(whichhascapacity[0][0])):
            
                        outlist[ whichexceed[0][1]] += int(np.round(whichhascapacity[0][0]))
                        outlist[ whichhascapacity[0][1]] -= int(np.round(whichhascapacity[0][0]))
                    else:
                        
                        outlist[ whichexceed[0][1]] += int(np.round(whichexceed[0][0]))
                        outlist[ whichhascapacity[0][1]] -= int(np.round(whichexceed[0][0]))
        #outspan = max(outlist) - min(outlist)

        probtracker.UpdateModPool(inlist, origoutlist, outlist)








    return outlist



def PureBHP(inputlist, argdict):
    alpha = 0
    return ModPool(inputlist, alpha)

def PureModPool(inputlist, argdict):
    alpha = 1
    return ModPool(inputlist, alpha)


      

def PureRandomWalkHeat(inlist, bContinuous = False):

    """
    Incoming particles perform random walks, with no ModPool or other variance reduction

    """

    if bContinuous:
        return ModPoolJustBrownian(inlist, True)

    brownian_pool = np.sum(inlist)


    # first, the Huygens portion 
    N_arc = len(inlist)

    outlist = np.zeros((N_arc,)).astype("int")

    pospool = np.sum([ np.max([x,0]) for x in inlist]) # np.max([incoming_along_xneg,0]) + np.max([incoming_along_yneg,0]) + np.max([incoming_along_xpos,0]) + np.max([incoming_along_ypos,0])
    negpool = np.sum([ np.min([x,0]) for x in inlist]) # np.min([incoming_along_xneg,0]) + np.min([incoming_along_yneg,0]) + np.min([incoming_along_xpos,0]) + np.min([incoming_along_ypos,0])

    posdeviate = np.random.multinomial(pospool, [1/float(N_arc)]*N_arc)
    negdeviate = np.random.multinomial(-negpool, [1/float(N_arc)]*N_arc) 
    for i in range(len(inlist)):
        outlist[i] = posdeviate[i] - negdeviate[i]

    #if np.sum(outlist) != 0:
    #    import pdb; pdb.set_trace()
    return outlist




def ModPoolJustBrownian(inlist, bContinuous = False):

    # ModPool, but without any Huygens step (and no multiplication by 2) -- i.e. the ModPool version of the heat equation 

    

    N_arc = len(inlist)
    brownian_pool = np.sum(inlist)

    if bContinuous:
        myres = np.ones(N_arc) * (brownian_pool / float(N_arc))
        return myres
        



    abspool = np.abs(brownian_pool)
    sgnpool = np.sign(brownian_pool)
    modpool = abspool % N_arc
    divpool = (abspool // N_arc) # this (times sgnpool) is a scalar value that gets added to every output arc

    NegateModpoolFactor = 1
    if modpool > N_arc // 2:
        NegateModpoolFactor = -1
        modpool = N_arc - modpool
        divpool += 1

    # this is the randomized "leftover" part
    #outdeviatemod = sgnpool * np.random.multinomial(modpool, [1/float(N_arc)]*N_arc)
    outdeviatemod = np.zeros((N_arc,)).astype("int")
    modpoolchoices = np.random.choice(N_arc, modpool, False, None)
    for i in modpoolchoices:
        outdeviatemod[i] += sgnpool * NegateModpoolFactor
    divpool *= sgnpool
    myres = outdeviatemod + divpool

    return myres


def ModPoolJustBrownianAmt(N_arc, Amt):

    # ModPool, but without any Huygens step (and no multiplication by 2) -- i.e. the ModPool version of the heat equation 
    brownian_pool = Amt

    abspool = np.abs(brownian_pool)
    sgnpool = np.sign(brownian_pool)
    modpool = abspool % N_arc
    divpool = (abspool // N_arc) # this (times sgnpool) is a scalar value that gets added to every output arc

    NegateModpoolFactor = 1
    if modpool > N_arc // 2:
        NegateModpoolFactor = -1
        modpool = N_arc - modpool
        divpool += 1

    # this is the randomized "leftover" part
    #outdeviatemod = sgnpool * np.random.multinomial(modpool, [1/float(N_arc)]*N_arc)
    outdeviatemod = np.zeros((N_arc,)).astype("int")
    modpoolchoices = np.random.choice(N_arc, modpool, False, None)
    for i in modpoolchoices:
        outdeviatemod[i] += sgnpool * NegateModpoolFactor
    divpool *= sgnpool
    myres = outdeviatemod + divpool

    return myres


    
# NOTE: This is only used to study  ZTokens in isolation.
def ZToken(inlist, argdict):

    """
    NOTE: This is only used to study  ZTokens in isolation. In the case of Fermi/Bose routines,
    it's easier just to use a variation of ModPool to update the ZTokens

    Same as ModPool but brownian pool is just sum(inlist), not twice that as it is in BHP
    """

    brownian_pool = np.sum(inlist)

    # first, the Huygens portion 
    N_arc = len(inlist)
    huygportion = copy(inlist)
    outlist = copy(inlist)
    for i in range(len(inlist)):
        huygportion[i] = -inlist[i]

        # in this case, we don't need a separate pospool and negpool, we just mod out the brownian (i.e. sum of neg and pos) pool
        abspool = np.abs(brownian_pool)
        sgnpool = np.sign(brownian_pool)
        modpool = abspool % N_arc
        divpool = (abspool // N_arc) # this (times sgnpool) is a scalar value that gets added to every output arc

        # if modpool is > N_arc//2, then we should change things so we only have to generate N_arc - modpool random variables.
        NegateModpoolFactor = 1
        if modpool > N_arc // 2:
            NegateModpoolFactor = -1
            modpool = N_arc - modpool
            divpool += 1

        # this is the randomized "leftover" part

        # this will work better for fermions in that the modpool portion will be distributed to different arcs
        # i.e. they will be spread out and never lump up
        outdeviatemod = np.zeros((N_arc,)).astype("int")
        modpoolchoices = np.random.choice(N_arc, modpool, False, None)
        for i in modpoolchoices:
            outdeviatemod[i] += sgnpool * NegateModpoolFactor

        divpool *= sgnpool

        for i in range(len(inlist)):
            outlist[i] = huygportion[i] + outdeviatemod[i] + divpool

    return outlist

 
# for alpha = 0, this generates pure B-H; for alpha = 1, we get modpool
def ZTokenCont(inlist, argdict):

    """
    Same as ModPool but brownian pool is just sum(inlist), not twice that as it is in BHP
    """

    avg_brownian_pool = np.mean(inlist)

    

    # first, the Huygens portion 
    N_arc = len(inlist)
    huygportion = copy(inlist)
    outlist = copy(inlist)
    for i in range(len(inlist)):
        outlist[i] = -inlist[i] + avg_brownian_pool

    #if np.sum(np.abs(inlist)) > 0:
    #    import pdb; pdb.set_trace()
    return outlist

def Bunch(inlist):

    """
    Identical to pure BHP except both Brownian particles emitted by any incoming
    particle get sent to same output arc.
    """

    brownian_pool = 2 * np.sum(inlist)


    # first, the Huygens portion 
    N_arc = len(inlist)

    huygportion = copy(inlist)
    outlist = copy(inlist)
    for i in range(len(inlist)):
        huygportion[i] = -inlist[i]




    pospool = np.sum([ np.max([x,0]) for x in inlist]) # np.max([incoming_along_xneg,0]) + np.max([incoming_along_yneg,0]) + np.max([incoming_along_xpos,0]) + np.max([incoming_along_ypos,0])
    negpool = np.sum([ np.min([x,0]) for x in inlist]) # np.min([incoming_along_xneg,0]) + np.min([incoming_along_yneg,0]) + np.min([incoming_along_xpos,0]) + np.min([incoming_along_ypos,0])

    posdeviate = np.random.multinomial(pospool, [1/float(N_arc)]*N_arc)
    negdeviate = np.random.multinomial(-negpool, [1/float(N_arc)]*N_arc) 
    for i in range(len(inlist)):
        outlist[i] = huygportion[i] + 2*posdeviate[i] - 2*negdeviate[i]

    return outlist


def trimfrac(negorposportion, Namt):
    """
    Remove Namt particles/antiparticles (note components of negorposportion are all of same sign or else zero)
    in proportin to their magnitudes. 
    """
    
    def getfraclist(xarr):
        if np.sum(xarr) == 0:
            print("ERROR in getfraclist() -- sum of input vector cannot be zero.")
        return np.array(xarr).astype("int") / float(Namt)

    fraclist = getfraclist(negorposportion)

    bNeg = np.sum(negorposportion) < 0
    outlist = np.zeros((len(negorposportion), )).astype("int")
    sumlaidoff = 0
    unusedprob = 0
    ind_lastunused = -1
    arglist = np.arange(len(negorposportion))
    rn.shuffle(arglist)
    for i in arglist:
        ifrac = fraclist[i]
        if ifrac == 0:
            continue
        if sumlaidoff > Namt:
            print("ERROR in trimfrac -- took off too much")
            import pdb; pdb.set_trace()
        elif sumlaidoff == Namt:
                outlist[i] = negorposportion[i]
                continue

        targetlayoff = Namt * np.abs(ifrac)
        roundedportion = int(np.trunc(targetlayoff))
        rounderr = targetlayoff - roundedportion
        padd1 = rounderr
        thisrand = rn.random()
        if padd1 + unusedprob > 1.0:
            unusedprob = padd1 + unusedprob - 1.0
            roundedportion += 1
            ind_lastunused = -1
        elif thisrand <= padd1 + unusedprob:
            roundedportion += 1
            unusedprob = 0
            ind_lastunused = -1
        else:
            unusedprob += rounderr
            ind_lastunused = copy(i)
        if roundedportion > 0:
            dock = np.max([0, np.min([Namt - sumlaidoff, roundedportion])])
            outlist[i] = int(np.sign(negorposportion[i])) * (np.abs(negorposportion[i]) - dock)
            sumlaidoff += dock
        else:
            outlist[i] = negorposportion[i] 

    
    NTries = 0
    if Namt - sumlaidoff > len(fraclist):
        print("WEIRD")
        import pdb; pdb.set_trace()
    if Namt - sumlaidoff > 1:
        print("ALSO WEIRD")
    
    if Namt - sumlaidoff == 1:
        if ind_lastunused < 0:
            print("problem: ind_lastunused should be in range 0...", len(inlist))
            import pdb; pdb.set_trace()
        outlist[ind_lastunused] += 1
        sumlaidoff += 1

    while sumlaidoff < Namt:
        print("I don't like this either.")
        import pdb; pdb.set_trace()
        shave = np.random.multinomial(1, fraclist)
        if bNeg:
            outlist = outlist - shave
        else:
            outlist = outlist + shave
        sumlaidoff += 1


    if sumlaidoff != Namt:
        print("ERROR in trimfrac")
        import pdb; pdb.set_trace()

    return outlist



def NetOut(inlist):
    return ModPoolJustBrownian(inlist)




def OnePlusBagOfOneAndNegativeOne(inlist):

    """
    instead of spawning a particle and antiparticle, input particle propagates SAME SIGN in backward direction, while
    emitting Brownian particle/antiparticle. This one is just for curiosity's sake.
    """

    


    # first, the Huygens portion 
    N_arc = len(inlist)

    huygportion = copy(inlist)
    outlist = copy(inlist)
    for i in range(len(inlist)):
        huygportion[i] = inlist[i]

    pospool = np.sum([ np.max([x,0]) for x in inlist]) # np.max([incoming_along_xneg,0]) + np.max([incoming_along_yneg,0]) + np.max([incoming_along_xpos,0]) + np.max([incoming_along_ypos,0])
    negpool = np.sum([ np.min([x,0]) for x in inlist]) # np.min([incoming_along_xneg,0]) + np.min([incoming_along_yneg,0]) + np.min([incoming_along_xpos,0]) + np.min([incoming_along_ypos,0])

    posdeviate = np.random.multinomial(pospool, [1/float(N_arc)]*N_arc)
    negdeviate = np.random.multinomial(-negpool, [1/float(N_arc)]*N_arc) 
    for i in range(len(inlist)):
        outlist[i] = huygportion[i] + posdeviate[i] - negdeviate[i]

    return outlist









def SplitPosNeg(inlist):
    N = len(inlist)
    poslist = np.zeros((N,)).astype("int")
    neglist = copy(poslist)

    for i in range(N):
        poslist[i] = np.max([0, inlist[i]])
        neglist[i] = np.min([0, inlist[i]])
    
    return poslist, neglist

def Reshape(xarr, shapefn):

    posxarr, negxarr = SplitPosNeg(xarr)
    posshape, negshape = SplitPosNeg(xarr)


    if np.sum(posxarr) > 0 and np.sum(posshape) > 0:
        shavedpos = trimfrac(posxarr, np.min([np.sum(posxarr), np.sum(posshape)]))
        outpos = posxarr - shavedpos
    else:
        outpos = posxarr

    if -np.sum(negxarr) > 0 and -np.sum(negshape) > 0:
        shavedneg = trimfrac(-negxarr, np.min([-np.sum(negxarr), -np.sum(negshape)]))
        outneg = negxarr + shavedneg
    else:
        outneg = negxarr
    
    return outpos + outneg



    

    

 
def OLDManageZ(incarr, incarr_br, argdict):
    global probtracker

    PARTICLE_LIMIT = argdict['Limit']

    N = len(incarr)
    sumin = np.sum(incarr)
    sumin_br = np.sum(incarr_br)

    #M = np.max((np.abs(sumin) // N, 1)) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    M = int(np.ceil(np.abs(sumin)/float(N))) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    abssum_br = np.abs(np.sum(incarr_br))
    M_br = int(np.ceil(abssum_br/float(N)))
    if M_br < 1:
        M_br = 1


    if np.max(np.abs(incarr_br)) <= M_br:
        # nothing needs to be done
        return incarr,  incarr_br
    if np.min(np.abs(incarr)) >= M:
        # nothing can be done
        return incarr,  incarr_br

    origincarr = copy(incarr)
    origincarr_br = copy(incarr_br)

    err = np.zeros(N).astype("int")
    
    upcapacity = np.zeros(N).astype("int")
    dncapacity = np.zeros(N).astype("int") 

    for i in range(N):
        iarr_br = incarr_br[i] 
        iarr = incarr[i]
        err[i] = np.max([0, iarr_br - M_br]) if iarr_br >= 0 else np.min([0, iarr_br + M_br])
        upcapacity[i] = np.max([np.min([M_br - iarr_br, M - iarr]), 0])
        dncapacity[i] = -np.max([np.min([M_br + iarr_br, M + iarr]), 0])


    origerr = copy(err) # get rid of this if you're not calling .updatemanagez()


    arng = np.arange(N)
    arngx = np.arange(N)
    np.random.shuffle(arng)
    np.random.shuffle(arngx) # probably could set arng to arngx, but will separate them anyway
    for iiarr in arng:
        ierr = err[iiarr]
        if ierr == 0:
            continue
        elif ierr > 0 and dncapacity[iiarr] < 0:
            for j in arngx:
                if j == iiarr:
                    continue
                if upcapacity[j] > 0:
                    thisamt = np.min([ierr, -dncapacity[iiarr], upcapacity[j]])
                    if thisamt == 0:
                        continue
                    ierr -= thisamt
                    #err[iiarr] -= thisamt
                    #iiarr index gets reduced
                    incarr[iiarr] -= thisamt
                    incarr_br[iiarr] -= thisamt
                    upcapacity[iiarr] = np.max([np.min([M_br - incarr_br[iiarr], M - incarr[iiarr]]), 0])
                    dncapacity[iiarr] = -np.max([np.min([M_br + incarr_br[iiarr], M + incarr_br[iiarr]]), 0])
                    err[iiarr] = np.max([0, incarr_br[iiarr]  - M_br]) if incarr_br[iiarr]  >= 0 else np.min([0, incarr_br[iiarr] + M_br])
                    #j index gets increased
                    incarr[j] += thisamt
                    incarr_br[j] += thisamt
                    upcapacity[j] = np.max([np.min([M_br - incarr_br[j], M - incarr[j]]), 0])
                    dncapacity[j] = -np.max([np.min([M_br + incarr_br[j], M + incarr_br[j]]), 0])
                    err[j] = np.max([0, incarr_br[j]  - M_br]) if incarr_br[j]  >= 0 else np.min([0, incarr_br[j] + M_br])

                    if ierr == 0:
                        break
        elif ierr < 0 and upcapacity[iiarr] > 0:
            for j in arngx:
                if j == iiarr:
                    continue
                if dncapacity[j] < 0:
                    thisamt = np.max([ierr, -upcapacity[iiarr], dncapacity[j]]) 
                    ierr -= thisamt
                    #err[iiarr] -= thisamt
                    if thisamt == 0:
                        continue
                    #iiarr index gets increased (since thisamt < 0)
                    incarr[iiarr] -= thisamt
                    incarr_br[iiarr] -= thisamt
                    upcapacity[iiarr] = np.max([np.min([M_br - incarr_br[iiarr], M - incarr[iiarr]]), 0])
                    dncapacity[iiarr] = -np.max([np.min([M_br + incarr_br[iiarr], M + incarr_br[iiarr]]), 0])
                    err[iiarr] = np.max([0, incarr_br[iiarr]  - M_br]) if incarr_br[iiarr]  >= 0 else np.min([0, incarr_br[iiarr] + M_br])
                    #j index gets reduced
                    incarr[j] += thisamt
                    incarr_br[j] += thisamt
                    upcapacity[j] = np.max([np.min([M_br - incarr_br[j], M - incarr[j]]), 0])
                    dncapacity[j] = -np.max([np.min([M_br + incarr_br[j], M + incarr_br[j]]), 0])
                    err[j] = np.max([0, incarr_br[j]  - M_br]) if incarr_br[j]  >= 0 else np.min([0, incarr_br[j] + M_br])

                    if ierr == 0:
                        break


    # either one of these will work...
    #probtracker.UpdateModPool(origincarr_br, origincarr_br, incarr_br)
    probtracker.UpdateManageZ(origerr, err)

    return incarr, incarr_br



 
 
def FermiOutM(incarr, argdict):
    global probtracker
    """
    Alternate way of doing BpseOutM that has its own internal
    FillIn function (and which is functionally very similar to
    how ManageZ routine works).
    """



    PARTICLE_LIMIT = argdict['Limit']


    M = PARTICLE_LIMIT # this is the pre-ModPool max-arc-count

    N = len(incarr)
    modpoolout = copy(incarr) # we can call it modpoolout since inputarcarr is the result of a ModPool calculation (see the calling function)

    if np.max(np.abs(modpoolout)) <= M:
        return modpoolout, np.zeros(N).astype("int")


    origincarr = copy(modpoolout)

    #err = copy(incarr_br) #np.zeros(N).astype("int")
    
    upcapacity = np.zeros(N).astype("int")
    dncapacity = np.zeros(N).astype("int") 

    for i in range(N):
        iarr = modpoolout[i]
        upcapacity[i] = np.max([M - iarr, 0])
        dncapacity[i] = -np.max([M + iarr, 0])


    sortind = []
    mu = np.mean(modpoolout)
    for iibr, ibr in enumerate(modpoolout):
        sortind.append((np.abs(ibr-mu) + (np.random.rand()-0.5) / 10.0, iibr))
    sortind.sort()
    sortind = [x[1] for x in sortind]
    sortbrind = sortind[::-1]
    
    # now, if we loop through sortbrind we will go from highest to lowest val of incarr_br (with sort-randomization of any identical values)

    for ii,i in enumerate(sortbrind):
        ibr = modpoolout[i]

        bAbove = ibr > mu
        if bAbove:
            if dncapacity[i] == 0:
                continue
        else:
            if upcapacity[i] == 0:
                continue

        for jj, j in enumerate(sortbrind):
            if ii == jj:
                continue
            jbr = modpoolout[j]
            if bAbove == (jbr > mu):
                continue

            
            if np.abs(ibr - jbr) <= 1: # note this is automatically true if i==j
                continue 
            if bAbove:

                if dncapacity[i] == 0:
                    continue
                # now, we can start to fill in
                thisamt = np.min([ (ibr - jbr)//2, -dncapacity[i], upcapacity[j]])
                
                modpoolout[i] -= thisamt
                modpoolout[j] += thisamt
                ibr = modpoolout[i]
                jbr = modpoolout[j]
                bAbove = ibr > mu

                upcapacity[i] = np.max([M - ibr, 0])
                dncapacity[i] = -np.max([M + ibr, 0])
                upcapacity[j] = np.max([M - jbr, 0])
                dncapacity[j] = -np.max([M + jbr, 0])
                if dncapacity[i] == 0:
                    break
            else:
                if upcapacity[i] == 0:
                    continue                
                thisamt = np.min([ np.max([0, jbr-M, -M-ibr]), -dncapacity[j], upcapacity[i]])
                modpoolout[j] -= thisamt
                modpoolout[i] += thisamt
                ibr = modpoolout[i]
                jbr = modpoolout[j]
                bAbove = ibr > mu

                upcapacity[i] = np.max([M - ibr, 0])
                dncapacity[i] = -np.max([M + ibr, 0])
                upcapacity[j] = np.max([M - jbr, 0])
                dncapacity[j] = -np.max([M + jbr, 0])
                if upcapacity[i] == 0:
                    break
                    
            if np.max(np.abs(modpoolout)) <= M:
                return modpoolout,  np.array(origincarr).astype("int") - np.array(modpoolout).astype("int")

    return modpoolout, np.array(origincarr).astype("int") - np.array(modpoolout).astype("int")

   
 
 
def ManageZ(incarr, incarr_br, argdict):
    global probtracker
    """
    In this simple version, we seek to minimize the sumsq or incarr_br by
    trying to shift down everything above np.mean(incarr_br)def i and shift up
    everything below, subject to the constraints that any shifts in incarr_br,
    when applied to incarr, do not induce a PEP violation in the latter.
    """


   

    PARTICLE_LIMIT = argdict['Limit']

    N = len(incarr)
    sumin = np.sum(incarr)
    sumin_br = np.sum(incarr_br)

    #M = np.max((np.abs(sumin) // N, 1)) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    M = int(np.ceil(np.abs(sumin)/float(N))) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    abssum_br = np.abs(np.sum(incarr_br))
    M_br = 0

    if np.max(np.abs(incarr_br)) == 0:
        # nothing needs to be done
        return incarr,  incarr_br

    origincarr = copy(incarr)
    origincarr_br = copy(incarr_br)

    #err = copy(incarr_br) #np.zeros(N).astype("int")
    
    upcapacity = np.zeros(N).astype("int")
    dncapacity = np.zeros(N).astype("int") 


    for i in range(N):
        iarr = incarr[i]
        upcapacity[i] = np.max([M - iarr, 0])
        dncapacity[i] = -np.max([M + iarr, 0])


    sortbrind = []
    for iibr, ibr in enumerate(incarr_br):
        sortbrind.append((ibr + (np.random.rand()-0.5) / 10.0, iibr))
    sortbrind.sort()
    sortbrind = [x[1] for x in sortbrind]
    sortbrind = sortbrind[::-1]
    

    bDone = False
    Niterations = -1
    

    while not(bDone):
        Niterations += 1
        bDidChange = False
        for ii, i in enumerate(sortbrind):
            ibr = incarr_br[i]
            
            for jj, j in enumerate(sortbrind):
                if ii == jj:
                    continue

                jbr = incarr_br[j]

                if ibr > jbr and (dncapacity[i] == 0 or upcapacity[j] == 0):
                    continue
                if ibr - jbr < 2:
                    continue




                # now, we can start to fill in
                thisamt = np.min([ (ibr - jbr)//2, -dncapacity[i], upcapacity[j]])
                incarr[i] -= thisamt
                incarr_br[i] -= thisamt
                incarr[j] += thisamt
                incarr_br[j] += thisamt
                ibr = incarr_br[i]
                jbr = incarr_br[j]

                upcapacity[i] = np.max([M - incarr[i], 0])
                dncapacity[i] = -np.max([M + incarr[i], 0])
                upcapacity[j] = np.max([M - incarr[j], 0])
                dncapacity[j] = -np.max([M + incarr[j], 0])
                bDidChange = True
        if not(bDidChange):
            bDone = True
        if Niterations > 5:
            print("Too many iterations")
            import pdb; pdb.set_trace()
          


    argdict["Reduction"] += np.sum(np.abs(origincarr_br)) - np.sum(np.abs(incarr_br))


    # Now, perform naive regularization of Z-tokens
    
    if argdict["Cutoff"] != 0:
        incarr_br_naivereg = Cutoff(incarr_br, argdict)
        return incarr, incarr_br_naivereg
    

    return incarr, incarr_br


   
 
def OLDManageZ(incarr, incarr_br, argdict):
    global probtracker




    def DiversifySorted(xarr):
        tuparr = []
        for iix, ix in enumerate(xarr):
            tuparr.append((np.abs(ix) + (np.random.rand()-0.5)/10, iix))
        
        tuparr.sort()
        tuparr = tuparr[::-1]
        
        return [x[1] for x in tuparr]
        
    

    PARTICLE_LIMIT = argdict['Limit']

    N = len(incarr)
    sumin = np.sum(incarr)
    sumin_br = np.sum(incarr_br)

    #M = np.max((np.abs(sumin) // N, 1)) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    M = int(np.ceil(np.abs(sumin)/float(N))) if PARTICLE_LIMIT == 0 else PARTICLE_LIMIT 
    abssum_br = np.abs(np.sum(incarr_br))
    M_br = 0

    if np.max(np.abs(incarr_br)) == 0:
        # nothing needs to be done
        return incarr,  incarr_br
    if np.min(np.abs(incarr)) >= M:
        # nothing can be done
        return incarr,  incarr_br

    origincarr = copy(incarr)
    origincarr_br = copy(incarr_br)

    err = copy(incarr_br) #np.zeros(N).astype("int")
    
    upcapacity = np.zeros(N).astype("int")
    dncapacity = np.zeros(N).astype("int") 


    for i in range(N):
        iarr_br = incarr_br[i] 
        iarr = incarr[i]
        #err[i] = np.max([0, iarr_br - M_br]) if iarr_br >= 0 else np.min([0, iarr_br + M_br])
        #upcapacity[i] = np.max([np.min([M_br - iarr_br, M - iarr]), 0])
        #dncapacity[i] = -np.max([np.min([M_br + iarr_br, M + iarr]), 0])

        upcapacity[i] = np.max([M - iarr, 0])
        dncapacity[i] = -np.max([M + iarr, 0])

    if np.max(upcapacity) == 0 or np.min(dncapacity) == 0:
        return incarr, incarr_br


    origerr = copy(err) # get rid of this if you're not calling .updatemanagez()


    arng = DiversifySorted(incarr_br) # np.arange(N)
    arngx = arng # np.arange(N)


    for iiarr in arng:
        ierr = err[iiarr]
        if ierr == 0:
            continue
        elif ierr > 0 and dncapacity[iiarr] < 0:
            for j in arngx:
                if j == iiarr:
                    continue
                # do not transfer a particle from one arc to another unless doing so brings down the magnitude of the max error
                if np.abs(ierr - err[j]) < 2:
                    continue


                if upcapacity[j] > 0:
                    thisamt = np.min([ierr, -dncapacity[iiarr], upcapacity[j]])
                    if thisamt == 0:
                        continue
                    ierr -= thisamt
                    #err[iiarr] -= thisamt
                    #iiarr index gets reduced
                    incarr[iiarr] -= thisamt
                    incarr_br[iiarr] -= thisamt
                    upcapacity[iiarr] = np.max([M - incarr[iiarr], 0])  # np.max([np.min([M_br - incarr_br[iiarr], M - incarr[iiarr]]), 0])
                    dncapacity[iiarr] = -np.max([M + incarr[iiarr], 0])  # -np.max([np.min([M_br + incarr_br[iiarr], M + incarr_br[iiarr]]), 0])
                    err[iiarr] -= thisamt # = np.max([0, incarr_br[iiarr]  - M_br]) if incarr_br[iiarr]  >= 0 else np.min([0, incarr_br[iiarr] + M_br])
                    #j index gets increased
                    incarr[j] += thisamt
                    incarr_br[j] += thisamt
                    upcapacity[j] = np.max([M - incarr[j], 0]) #np.max([np.min([M_br - incarr_br[j], M - incarr[j]]), 0])
                    dncapacity[j] = -np.max([M + incarr[j], 0])  #-np.max([np.min([M_br + incarr_br[j], M + incarr_br[j]]), 0])
                    err[j] = incarr_br[j] # np.max([0, incarr_br[j]  - M_br]) if incarr_br[j]  >= 0 else np.min([0, incarr_br[j] + M_br])
                    if ierr == 0:
                        break
        elif ierr < 0 and upcapacity[iiarr] > 0:
            for j in arngx:
                if j == iiarr:
                    continue
                # do not transfer a particle from one arc to another unless doing so brings down the magnitude of the max error
                if np.abs(ierr - err[j]) < 2:
                    continue
                if dncapacity[j] < 0:
                    thisamt = np.max([ierr, -upcapacity[iiarr], dncapacity[j]]) 
                    ierr -= thisamt
                    #err[iiarr] -= thisamt
                    if thisamt == 0:
                        continue
                    #iiarr index gets increased (since thisamt < 0)
                    incarr[iiarr] -= thisamt
                    incarr_br[iiarr] -= thisamt
                    upcapacity[iiarr] = np.max([M - incarr[iiarr], 0])  # np.max([np.min([M_br - incarr_br[iiarr], M - incarr[iiarr]]), 0])
                    dncapacity[iiarr] = -np.max([M + incarr[iiarr], 0])  # -np.max([np.min([M_br + incarr_br[iiarr], M + incarr_br[iiarr]]), 0])
                    err[iiarr] -= thisamt # = np.max([0, incarr_br[iiarr]  - M_br]) if incarr_br[iiarr]  >= 0 else np.min([0, incarr_br[iiarr] + M_br])
                    #j index gets reduced
                    incarr[j] += thisamt
                    incarr_br[j] += thisamt
                    upcapacity[j] = np.max([M - incarr[j], 0]) #np.max([np.min([M_br - incarr_br[j], M - incarr[j]]), 0])
                    dncapacity[j] = -np.max([M + incarr[j], 0])  #-np.max([np.min([M_br + incarr_br[j], M + incarr_br[j]]), 0])
                    err[j] = incarr_br[j] # np.max([0, incarr_br[j]  - M_br]) if incarr_br[j]  >= 0 else np.min([0, incarr_br[j] + M_br])

                    if ierr == 0:
                        break

    probtracker.UpdateManageZ(origerr, err)



    return incarr, incarr_br


   


def PedanticKeepGoing(incarr, incarr_br, argdict): # PairTokenThenSubtract(incarr, incarr_br, CoreFunction):
    """

    This version keeps whorl segments until it meets an opposite end (i.e. no attempt to
    convert incomplete whorl segments into something that has the right wave statistics). 
    Even if it encounters a node where everything is zero (or even flat), the routine 
    in that case BUILDS the whorl until it finds an opposing  filament. 

    It is assumed that this version therefore explodes the Nabs initially until the space is packed with whorls,
    at which point finding an opposing whorl segment can be done easily. 
    Ideally, it leads to a situation where every element  of space is connected by a whorl segment 
    to every other element of space, creating an immensely deep sea of whorls, but once that's done, 
    any further growth will be unnecessary. That's the motivation, anyway. 


    incarr_br: OLD (incoming tokens)
    forward_emission_heat_bath_contribution_huygens: Suriviving old incoming tokens

    """


    
    CoreFunction = argdict['CoreFn']


    forward_emission_heat_bath_contribution_huygens = ModPoolJustBrownian(incarr_br) # the reverse of this gets added in to the regular particles

    #unmatched_incarr_br = FillIn(incarr_br) # this can be regarded as the portion of incarr_br that doesn't get wiped out in the  ModPoolJustBrownian() routine

    # first, make sure all the z-tokens (contained in incarr_br) are "loaded" with a partner regular particle so as to render them into true whorls
    newincarr = incarr - np.array(incarr_br).astype("int") 

    reg_particle_incoming = newincarr  + forward_emission_heat_bath_contribution_huygens # it is positive because you negate displaced below...

    mp = ModPool(reg_particle_incoming)

    brownhuygout, displaced = CoreFunction(mp, argdict)


    if CoreFunction == FermiOut6 and np.max(np.abs(brownhuygout)) > 1:
        print("violated PEP")
        import pdb; pdb.set_trace()



    if np.sum(incarr) != np.sum(brownhuygout):
        print("violated amplitude")
        import pdb; pdb.set_trace()


    
    if CoreFunction  == FermiOutM and np.max(np.abs(brownhuygout)) > argdict['Limit']:
        print("violated PEP")        
        import pdb; pdb.set_trace()


    bManageZTokens = False
    if bManageZTokens:
        origbrown = copy(brownhuygout)



        newbrown, newdisp = ManageZ(brownhuygout, -displaced  + forward_emission_heat_bath_contribution_huygens, argdict)
        if np.max(np.abs(newbrown)) > argdict['Limit']:
            print("violated PEP (out)")
            import pdb; pdb.set_trace()


        
        return newbrown, newdisp  #+ forward_emission_heat_bath_contribution_huygens

    #return brownhuygout + forward_emission_heat_bath_contribution_huygens, -displaced  + forward_emission_heat_bath_contribution_huygens
    return brownhuygout, -displaced  + forward_emission_heat_bath_contribution_huygens



    


def FermiOut6(inputarcarr, argdict):
    """
    In this version we will calculate the modpool output and then calc a deviation-from-PEP vector, and then
    a displacement vector (whose sum is zero) to make the deviation-from-PEP zero.
    """

    # deprecated
    def WhicheverWorks(err, modpoolout, signsumerr):
        """
        Switched to using this because it "spreads" the possible return args
        better than WhichHaveMostOppositeComp; to see why consider a case
        where one coefficient is huge and of opposite sign. That (or some
        proportional thing so that if it's huge and of opposite sign it gets
        chosen most of the time) will preferentially return one index, whereas
        this version will scramble things around better.

        """

        #import pdb; pdb.set_trace()
        capacity = []

        for i in range(len(err)):
            if np.abs(modpoolout[i] + signsumerr) <= 1:
                capacity.append(i)
        
        if len(capacity) == 0:
            import pdb; pdb.set_trace()

        return np.random.choice(capacity)

    def WhichHaveMostOppositeComp(err, modpoolout, signsumerr):

        #import pdb; pdb.set_trace()
        capacity = np.zeros((len(err),)).astype("int")
        for i in range(len(err)):
            if err[i] * signsumerr <= 0:
                capacity[i] = (modpoolout[i] * signsumerr - 1)


        minlevel = np.min(capacity)
        if minlevel == 0:
            return None
        if minlevel > 0:
            print("no more offloading possible:", inputarcarr)
            import pdb; pdb.set_trace()
        whichhavemin = []
        for jjerr, jerr in enumerate(capacity):
            if jerr == minlevel:
                whichhavemin.append(jjerr)
        #import pdb; pdb.set_trace()
        return np.random.choice(whichhavemin)

    # first, take care of the Ztokens (which used to be referred to as just a "heat bath")

    if np.abs(np.sum(inputarcarr)) > len(inputarcarr):
        print("Error in FermiOut6() -- inputs violate PEP so that PEP cannot be satisfied.")
        import pdb; pdb.set_trace()




    bPrint = False
    #if np.sum(inputarcarr) > 3:
    #    bPrint = True



    N = len(inputarcarr)

    modpoolout = copy(inputarcarr) # this misnomer has been preserved given that inputarcarr has indeed been modpooled in the calling module
    origmodpoolout = np.copy(modpoolout)
    if np.max(np.abs(modpoolout)) <= 1:
        if bPrint:
            print("out", inputarcarr, modpoolout)
        return modpoolout,  np.zeros((N, )).astype("int")

    err = np.zeros((N, )).astype("int")

    for iim, im in  enumerate(modpoolout):
        if np.abs(im) > 1:
            err[iim] = im - np.sign(im)
            modpoolout[iim] = np.sign(im)

    #if tuple(inputarcarr) == (-1,  0,  1,  1):
    #    print("na")
    #    import pdb; pdb.set_trace()


    sumerr = np.sum(err)
    signsumerr = int(np.sign(sumerr))


    if sumerr == 0:
        return modpoolout, err

    # we will simply "drip" the error away, always shifting the excess bits to indices 
    # with the most capacity (i.e. whose values have the opposite sign of sumerr)

    #import pdb; pdb.set_trace()
    for i in range(np.abs(sumerr)):

        # consider switching out the following two lines of code
        whichind = WhichHaveMostOppositeComp(err, modpoolout, signsumerr)
        #whichind = WhicheverWorks(err, modpoolout, signsumerr)
        
        if whichind is None:
            print("ERROR in FermiOut6: cannot lay off all the excess; consider changing whichind assignment")
            import pdb; pdb.set_trace()
        err[whichind] -= signsumerr
        modpoolout[whichind] += signsumerr


    if np.max(np.abs(modpoolout)) > 1:
        print("ERROR in FermiOut6: could not lay off excess; PEP violation")
        import pdb; pdb.set_trace()

    return modpoolout, err


   

def FindNonZeroCoords(ggraph):
    for i in range(len(ggraph.NodeVec)):
        if np.sum(np.abs(ggraph.NodeVec[i].Amplitude)) != 0:
            print (ggraph.NodeVec[i].Coords)
            

def FindNonZeroCoordsCtl(ggraph):
    for i in range(len(ggraph.NodeVec)):
        if np.sum(np.abs(ggraph.NodeVec[i].AmplitudeCtl)) != 0:
            print (ggraph.NodeVec[i].Coords)

def FindNonZeroCoordsZ(ggraph):
    for i in range(len(ggraph.NodeVec)):
        if np.sum(np.abs(ggraph.NodeVec[i].ZAmplitude)) != 0:
            print (ggraph.NodeVec[i].Coords)
            

def FindNonZeroCoordsScr(ggraph):
    for i in range(len(ggraph.NodeVec)):
        if np.sum(np.abs(ggraph.NodeVec[i].Scratch)) != 0:
            print (ggraph.NodeVec[i].Coords)
            

def FindNonZeroCoordsScrCtl(ggraph):
    for i in range(len(ggraph.NodeVec)):
        if np.sum(np.abs(ggraph.NodeVec[i].ScratchCtl)) != 0:
            print (ggraph.NodeVec[i].Coords)
            



    






# only works for a 4x4 (i.e. 2-dim) grid, but is useful for debugging small routines
# and with minor modification can be used to nicely print out 4x4 submatrices of larger matrices
def print4grid(grid, t):

    # old 0 --> new 0
    # old 1 --> new 2
    # old 2 --> new 1
    # old 3 --> new 3

    def chstr(xstr, pos, c):
        mylist = list(xstr)
        mylist[pos] = c
        return ''.join(mylist)
    spc = 6
    vrt = 4

    #import pdb; pdb.set_trace()
    flines = [] 
    #              sum(0,0)                x+                  sum(0,1)                    x-       
    flines.append(('_' * spc + '=' * spc + '_' * spc + '=' * spc + '_' * spc + '=' * spc + '_' * spc + '=' * spc)*2)
    for i in range(vrt):
        flines.append(('  ||  ' + ' ' * 3 * spc +            '  ||  '    + ' ' * spc + ' ' * spc + ' ' * spc)*2)
    flines.append(('_' * spc + ' ' * spc + ' ' * spc  + ' ' * spc + '_' * spc + ' ' * spc * 3 )*2) # y- and y+
    for i in range(vrt):
        flines.append(('  ||  ' + ' ' * 3 * spc +            '  ||  '    + ' ' * spc + ' ' * spc + ' ' * spc) *2)
    flines.append(('_' * spc + '=' * spc + '_' * spc + '=' * spc + '_' * spc + '=' * spc + '_' * spc + '=' * spc)*2) # sum(0,1) and sum(1,1)
    for i in range(vrt):
        flines.append(('  ||  ' + ' ' * 3 * spc  +          '  ||  '    + ' ' * spc + ' ' * spc + ' ' * spc)*2)
    flines.append(('_' * spc + ' ' * spc + ' ' * spc  + ' ' * spc + '_' * spc + ' ' * spc * 3 )*2) #  # y- and y+
    for i in range(vrt):
        flines.append(('  ||  ' + ' ' * 3 * spc  +          '  ||  '    + ' ' * spc + ' ' * spc + ' ' * spc)*2)
    
    flines = flines + copy(flines)

    
    
    if t % 2 == 0:
        flines[0] = chstr(flines[0], 4*spc-1, '>')
        flines[0] = chstr(flines[0], 5*spc, '<')
        flines[0] = chstr(flines[0], 12*spc-1, '>')
        flines[0] = chstr(flines[0], 13*spc, '<')



        flines[1] = chstr(flines[1], 4*spc+2, '/')
        flines[1] = chstr(flines[1], 4*spc+3, '\\')        
        flines[1] = chstr(flines[1], 12*spc+2, '/')
        flines[1] = chstr(flines[1], 12*spc+3, '\\')

        flines[2*vrt+1] = chstr(flines[2*vrt+1], 2, '\\')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 3, '/')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 8*spc+2, '\\')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 8*spc+3, '/')



        flines[2*vrt+3] = chstr(flines[2*vrt+3], 2, '/')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 3, '\\')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 8*spc+2, '/')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 8*spc+3, '\\')



        #import pdb; pdb.set_trace()
        flines[2*vrt+2] = chstr(flines[2*vrt+2], spc, '<')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], -1, '>')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], 8*spc-1, '>')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], 9*spc, '<')

        flines[4*vrt+4] = chstr(flines[4*vrt+4], 4*spc-1, '>')
        flines[4*vrt+4] = chstr(flines[4*vrt+4], 5*spc, '<')
        flines[4*vrt+4] = chstr(flines[4*vrt+4], 12*spc-1, '>')
        flines[4*vrt+4] = chstr(flines[4*vrt+4], 13*spc, '<')

        
        flines[6*vrt+6] = chstr(flines[6*vrt+6], spc, '<')
        flines[6*vrt+6] = chstr(flines[6*vrt+6], -1, '>')
        flines[6*vrt+6] = chstr(flines[6*vrt+6], 8*spc-1, '>')
        flines[6*vrt+6] = chstr(flines[6*vrt+6], 8*spc+spc, '<')
      
        flines[6*vrt+5] = chstr(flines[6*vrt+5], 2, '\\')
        flines[6*vrt+5] = chstr(flines[6*vrt+5], 3, '/')
        flines[6*vrt+5] = chstr(flines[6*vrt+5], 8*spc+2, '\\')
        flines[6*vrt+5] = chstr(flines[6*vrt+5], 8*spc+3, '/')

        flines[6*vrt+7] = chstr(flines[6*vrt+7], 2, '/')
        flines[6*vrt+7] = chstr(flines[6*vrt+7], 3, '\\')
        flines[6*vrt+7] = chstr(flines[6*vrt+7], 8*spc+2, '/')
        flines[6*vrt+7] = chstr(flines[6*vrt+7], 8*spc+3, '\\')




        hlfwy = len(flines) // 2
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 4*spc+2, '/')
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 4*spc+3, '\\')        
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 12*spc+2, '/')
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 12*spc+3, '\\')

        flines[hlfwy-1] = chstr(flines[hlfwy-1], 4*spc+2, '\\')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 4*spc+3, '/')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 12*spc+2, '\\')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 12*spc+3, '/')




        flines[-1] = chstr(flines[-1], 4*spc+2, '\\')
        flines[-1] = chstr(flines[-1], 4*spc+3, '/')
        flines[-1] = chstr(flines[-1], 12*spc+2, '\\')
        flines[-1] = chstr(flines[-1], 12*spc+3, '/')

        #import pdb; pdb.set_trace()
        a = list(flines[0])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(np.sum(grid[0,0,:])) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[0,0,0])
        a[2*spc:3*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[2,0,1])
        a[6*spc:7*spc] = list(b)
        #quadrant2,0
        b = '{:^6d}'.format(np.sum(grid[2,0,:])) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[2,0,0])
        a[10*spc:11*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[0,0,1])
        a[14*spc:15*spc] = list(b)
        flines[0] = ''.join(a)

        a = list(flines[vrt+1])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,0,2]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,1,3]) 
        a[4*spc:5*spc] = list( b)
        #quadrant2,1
        b = '{:^6d}'.format(grid[2,0,2]) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[3,1,3]) 
        a[12*spc:13*spc] = list( b)
        flines[vrt+1] = ''.join(a)



        a = list(flines[2*vrt+2])
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,1,1])
        a[2*spc:3*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[1,1,:])) 
        a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[1,1,0])
        a[6*spc:7*spc] = list(b)
        #quadrant2,2
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[3,1,1])
        a[10*spc:11*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[3,1,:])) 
        a[12*spc:13*spc] = list( b )
        b = '{:^6d}'.format(grid[3,1,0])
        a[14*spc:15*spc] = list(b)
        flines[2*vrt+2] = ''.join(a)

        
        a = list(flines[3*vrt+3])
        b = '{:^6d}'.format(grid[0,2,3]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,1,2]) 
        a[4*spc:5*spc] = list( b)
        #quadrant2,3
        b = '{:^6d}'.format(grid[2,2,3]) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[3,1,2]) 
        a[12*spc:13*spc] = list( b)
        flines[3*vrt+3] = ''.join(a)
        #import pdb; pdb.set_trace()


        # 2nd half
        hlfwy = len(flines) // 2
    
        a = list(flines[hlfwy+0])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(np.sum(grid[0,2,:])) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[0,2,0])
        a[2*spc:3*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[2,2,1])
        a[6*spc:7*spc] = list(b)
        #quadrant2,0
        b = '{:^6d}'.format(np.sum(grid[2,2,:])) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[2,2,0])
        a[10*spc:11*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[0,2,1])
        a[14*spc:15*spc] = list(b)
        flines[hlfwy+0] = ''.join(a)

        a = list(flines[hlfwy+vrt+1])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,2,2]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,3,3]) 
        a[4*spc:5*spc] = list( b)
        #quadrant2,1
        b = '{:^6d}'.format(grid[2,2,2]) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[3,3,3]) 
        a[12*spc:13*spc] = list( b)
        flines[hlfwy+vrt+1] = ''.join(a)



        a = list(flines[hlfwy+2*vrt+2])
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,3,1])
        a[2*spc:3*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[1,3,:])) 
        a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[1,3,0])
        a[6*spc:7*spc] = list(b)
        #quadrant2,2
        #b = '{:^6d}'.format(np.sum(grid[0,0,1,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[3,3,1])
        a[10*spc:11*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[3,3,:])) 
        a[12*spc:13*spc] = list( b )
        b = '{:^6d}'.format(grid[3,3,0])
        a[14*spc:15*spc] = list(b)
        flines[hlfwy+2*vrt+2] = ''.join(a)


        a = list(flines[hlfwy+3*vrt+3])
        b = '{:^6d}'.format(grid[0,0,3]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,3,2]) 
        a[4*spc:5*spc] = list( b)
        #quadrant2,3
        b = '{:^6d}'.format(grid[2,0,3]) 
        a[8*spc:9*spc] = list( b )
        b = '{:^6d}'.format(grid[3,3,2]) 
        a[12*spc:13*spc] = list( b)
        flines[hlfwy+3*vrt+3] = ''.join(a)




    if t % 2 == 1:
        #import pdb; pdb.set_trace()

        flines[0] = chstr(flines[0], spc, '<')
        flines[0] = chstr(flines[0], -1, '>')
        flines[0] = chstr(flines[0], 8*spc-1, '>')
        flines[0] = chstr(flines[0], 9*spc, '<')


        flines[1] = chstr(flines[1], 2, '/')
        flines[1] = chstr(flines[1], 3, '\\')
        flines[1] = chstr(flines[1], 8*spc+2, '/')
        flines[1] = chstr(flines[1], 8*spc+3, '\\')
        
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 4*spc + 2, '\\')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 4*spc + 3, '/')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 12*spc + 2, '\\')
        flines[2*vrt+1] = chstr(flines[2*vrt+1], 12*spc + 3, '/')

        flines[2*vrt+3] = chstr(flines[2*vrt+3], 4*spc + 2, '/')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 4*spc + 3, '\\')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 12*spc + 2, '/')
        flines[2*vrt+3] = chstr(flines[2*vrt+3], 12*spc + 3, '\\')


        flines[2*vrt+2] = chstr(flines[2*vrt+2], 4*spc-1, '>')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], 5*spc, '<')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], 12*spc-1, '>')
        flines[2*vrt+2] = chstr(flines[2*vrt+2], 13*spc, '<')


        #???
        hlfwy = len(flines)//2

        flines[hlfwy+0] = chstr(flines[hlfwy+0], spc, '<')
        flines[hlfwy+0] = chstr(flines[hlfwy+0], -1, '>')
        flines[hlfwy+0] = chstr(flines[hlfwy+0], 8*spc-1, '>')
        flines[hlfwy+0] = chstr(flines[hlfwy+0], 9*spc, '<')


        flines[hlfwy+1] = chstr(flines[hlfwy+1], 2, '/')
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 3, '\\')
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 8*spc+2, '/')
        flines[hlfwy+1] = chstr(flines[hlfwy+1], 8*spc+3, '\\')
        
        flines[hlfwy+2*vrt+1] = chstr(flines[hlfwy+2*vrt+1], 4*spc + 2, '\\')
        flines[hlfwy+2*vrt+1] = chstr(flines[hlfwy+2*vrt+1], 4*spc + 3, '/')
        flines[hlfwy+2*vrt+1] = chstr(flines[hlfwy+2*vrt+1], 12*spc + 2, '\\')
        flines[hlfwy+2*vrt+1] = chstr(flines[hlfwy+2*vrt+1], 12*spc + 3, '/')

        flines[hlfwy+2*vrt+3] = chstr(flines[hlfwy+2*vrt+3], 4*spc + 2, '/')
        flines[hlfwy+2*vrt+3] = chstr(flines[hlfwy+2*vrt+3], 4*spc + 3, '\\')
        flines[hlfwy+2*vrt+3] = chstr(flines[hlfwy+2*vrt+3], 12*spc + 2, '/')
        flines[hlfwy+2*vrt+3] = chstr(flines[hlfwy+2*vrt+3], 12*spc + 3, '\\')


        flines[hlfwy+2*vrt+2] = chstr(flines[hlfwy+2*vrt+2], 4*spc-1, '>')
        flines[hlfwy+2*vrt+2] = chstr(flines[hlfwy+2*vrt+2], 5*spc, '<')
        flines[hlfwy+2*vrt+2] = chstr(flines[hlfwy+2*vrt+2], 12*spc-1, '>')
        flines[hlfwy+2*vrt+2] = chstr(flines[hlfwy+2*vrt+2], 13*spc, '<')



        flines[hlfwy-1] = chstr(flines[hlfwy-1], 2, '\\')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 3, '/')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 8*spc+2, '\\')
        flines[hlfwy-1] = chstr(flines[hlfwy-1], 8*spc+3, '/')


        flines[-1] = chstr(flines[-1], 2, '\\')
        flines[-1] = chstr(flines[-1], 3, '/')
        flines[-1] = chstr(flines[-1], 8*spc+2, '\\')
        flines[-1] = chstr(flines[-1], 8*spc+3, '/')




        a = list(flines[0])
        #import pdb; pdb.set_trace()
        #b = '{:^6d}'.format(np.sum(grid[0,0,0,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,0,1])
        a[2*spc:3*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[1,0,:])) 
        a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[1,0,0])
        a[6*spc:7*spc] = list(b)

        b = '{:^6d}'.format(grid[3,0,1])
        a[10*spc:11*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[3,0,:])) 
        a[12*spc:13*spc] = list( b )
        b = '{:^6d}'.format(grid[3,0,0])


        a[14*spc:15*spc] = list(b)

        flines[0] = ''.join(a)


        #import pdb; pdb.set_trace()
        a = list(flines[vrt+1])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,1,3]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,0,2]) 
        a[4*spc:5*spc] = list( b)

        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[2,1,3]) 
        a[(8*spc+0):(8*spc+spc)] = list( b )
        b = '{:^6d}'.format(grid[3,0,2]) 
        a[(8*spc+4*spc):(8*spc+5*spc)] = list( b)


        flines[vrt+1] = ''.join(a)

        

        a = list(flines[2*vrt+2])
        b = '{:^6d}'.format(np.sum(grid[0,1,:])) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[0,1,0])
        a[2*spc:3*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,1,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[0,1,1])
        a[14*spc:15*spc] = list(b)

        b = '{:^6d}'.format(np.sum(grid[2,1,:]))
        a[(8*spc+0):(8*spc+spc)] = list( b )
        b = '{:^6d}'.format(grid[2,1,0])
        a[(8*spc + 2*spc):(8*spc + 3*spc)] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,1,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[2,1,1])
        a[6*spc:7*spc] = list(b)

        flines[2*vrt+2] = ''.join(a)



        a = list(flines[3*vrt+3])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,1,2]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,2,3]) 
        a[4*spc:5*spc] = list( b)

        b = '{:^6d}'.format(grid[2,1,2]) 
        a[(8*spc + 0*spc):(8*spc + 1*spc)] = list( b )
        b = '{:^6d}'.format(grid[3,2,3]) 
        a[(8*spc + 4*spc):(8*spc + 5*spc)] = list( b)


        flines[3*vrt+3] = ''.join(a)




        # lower half


        a = list(flines[hlfwy+0])
        #import pdb; pdb.set_trace()
        #b = '{:^6d}'.format(np.sum(grid[0,0,0,:])) 
        #a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,2,1])
        a[2*spc:3*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[1,2,:])) 
        a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[1,2,0])
        a[6*spc:7*spc] = list(b)

        b = '{:^6d}'.format(grid[3,2,1])
        a[10*spc:11*spc] = list( b)
        b = '{:^6d}'.format(np.sum(grid[3,2,:])) 
        a[12*spc:13*spc] = list( b )
        b = '{:^6d}'.format(grid[3,2,0])


        a[14*spc:15*spc] = list(b)

        flines[hlfwy+0] = ''.join(a)


        #import pdb; pdb.set_trace()
        a = list(flines[hlfwy+vrt+1])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,3,3]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,2,2]) 
        a[4*spc:5*spc] = list( b)

        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[2,3,3]) 
        a[(8*spc+0):(8*spc+spc)] = list( b )
        b = '{:^6d}'.format(grid[3,2,2]) 
        a[(8*spc+4*spc):(8*spc+5*spc)] = list( b)


        flines[hlfwy+vrt+1] = ''.join(a)

        

        a = list(flines[hlfwy+2*vrt+2])
        b = '{:^6d}'.format(np.sum(grid[0,3,:])) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[0,3,0])
        a[2*spc:3*spc] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,1,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[2,3,1])
        a[6*spc:7*spc] = list(b)

        b = '{:^6d}'.format(np.sum(grid[2,3,:]))
        a[(8*spc+0):(8*spc+spc)] = list( b )
        b = '{:^6d}'.format(grid[2,3,0])
        a[(8*spc + 2*spc):(8*spc + 3*spc)] = list( b)
        #b = '{:^6d}'.format(np.sum(grid[0,1,1,:])) 
        #a[4*spc:5*spc] = list( b )
        b = '{:^6d}'.format(grid[0,3,1])
        a[14*spc:15*spc] = list(b)

        flines[hlfwy+2*vrt+2] = ''.join(a)



        a = list(flines[hlfwy+3*vrt+3])
        #import pdb; pdb.set_trace()
        b = '{:^6d}'.format(grid[0,3,2]) 
        a[0:spc] = list( b )
        b = '{:^6d}'.format(grid[1,0,3]) 
        a[4*spc:5*spc] = list( b)

        b = '{:^6d}'.format(grid[2,3,2]) 
        a[(8*spc + 0*spc):(8*spc + 1*spc)] = list( b )
        b = '{:^6d}'.format(grid[3,0,3]) 
        a[(8*spc + 4*spc):(8*spc + 5*spc)] = list( b)


        flines[hlfwy+3*vrt+3] = ''.join(a)



    for il in flines:
       print(il)
    #import pdb; pdb.set_trace()
      






def Iterate(ggraph, opts, argdict, bContinuous=False):
    bPedantic = True
    if opts.zdynamics[:3] == 'alt':
        bPedantic = False

    InOutFunction = argdict['InOutFunction']
    CoreFn = argdict['CoreFn']


    arngshuff = np.arange(ggraph.NNode)

    bShuffleAtEachStep = True
    if bShuffleAtEachStep:
        rn.shuffle(arngshuff)

    previ = -1
    
    for i in arngshuff:
        inlist = []
        if ggraph.bNeedZTokens:
            inlist_ztoken = []



        inlist_ctl = []
        for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
            whichnbr = ggraph.NodeVec[i].FromNbr[iinbr]   

            inlist.append(ggraph.NodeVec[inbr].Amplitude[whichnbr])

            if not(argdict["CoreFn"] in (ModPoolJustBrownian, PureRandomWalkHeat)):
                inlist_ctl.append(ggraph.NodeVec[inbr].AmplitudeCtl[whichnbr])

            if ggraph.bNeedZTokens:
                inlist_ztoken.append(ggraph.NodeVec[inbr].ZAmplitude[whichnbr]) 


        previ = copy(i)

        if  bContinuous:
            inlist = []
            
            for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
                whichnbr = ggraph.NodeVec[i].FromNbr[iinbr]            
                inlist.append(ggraph.NodeVec[inbr].ContAmplitude[whichnbr])
            if argdict["CoreFn"] in (ModPoolJustBrownian, PureRandomWalkHeat):
                floatingoutamp = ModPoolJustBrownian(inlist, True)
            else:
                floatingoutamp = FloatingPtWave(inlist)
            ggraph.NodeVec[i].ContScratch = floatingoutamp
        else:
            if argdict["InOutFunction"] != ModPoolJustBrownian:
                outamp_ctl = ModPool(inlist_ctl)  
                #comparison = outamp_ctl
                #incomparison = inlist_ctl

            if ggraph.bNeedZTokens:
                ggraph.NodeVec[i].ZScratchIn = inlist_ztoken               
                outamp, outampz = InOutFunction(inlist, inlist_ztoken, argdict)    


              
            
            else:
                outamp = InOutFunction(inlist)       
                outamp_ctl = outamp


            ggraph.NodeVec[i].PrevAmplitude = copy(ggraph.NodeVec[i].Amplitude)
            ggraph.NodeVec[i].Scratch = outamp
            ggraph.NodeVec[i].ScratchCtl = outamp_ctl

            
            if ggraph.bNeedZTokens:
                ggraph.NodeVec[i].ZScratch = outampz
                

        
    
    if  bContinuous:
        
        for i in arngshuff:
            ggraph.NodeVec[i].ContAmplitude = ggraph.NodeVec[i].ContScratch
    else:
        for i in arngshuff:
            ggraph.NodeVec[i].Amplitude = ggraph.NodeVec[i].Scratch
        
            ggraph.NodeVec[i].AmplitudeCtl = ggraph.NodeVec[i].ScratchCtl
            if ggraph.bNeedZTokens:                

                # Note that when exporting to an array (incase of a D-dimensional array setup) 
                # there needs to be an offsetting Z-amp update at the DESTINATION of the outgoing arc
                ggraph.NodeVec[i].ZAmplitude = ggraph.NodeVec[i].ZScratch
                #ggraph.NodeVec[i].ZAmplitudeIn = ggraph.NodeVec[i].ZScratchIn
    
         
 
def Cutoff(inlist, argdict):
    CUTOFF = argdict["Cutoff"]
    CUTOFFVOLUME = argdict["CutoffVolume"]

    maxoutgoing = np.max(inlist)
    minoutgoing = np.min(inlist)

    N_arc = len(inlist)

    outlist = copy(inlist)


    if maxoutgoing <= CUTOFF and -minoutgoing <= CUTOFF:
        pass

    elif maxoutgoing - CUTOFF >= (-minoutgoing - CUTOFF):
        whichexceed = []
        sumexcess = 0
        whichhascapacity = []
        sumcapacity = 0
        for i in range(N_arc):
            excess = np.max([0, outlist[i] - CUTOFF])

            
            if CUTOFFVOLUME != 0:
                excess = np.min([CUTOFFVOLUME, excess])
            if excess > 0:
                whichexceed.append((excess + np.random.normal()/100.0, i)) #
                sumexcess += excess

            capacity = np.min([0, outlist[i]])
            if capacity < 0:
                whichhascapacity.append((capacity + np.random.normal()/100.0, i))
                sumcapacity += capacity

        whichexceed.sort()
        whichexceed.reverse() 
        whichhascapacity.sort()

        # let's start with the simplest choice (more or less) of just annihilating the biggest excsss with the biggest capacity
        if rn.random() < FAC:
            if sumcapacity < 0 and sumexcess > 0:
                if int(np.round(whichexceed[0][0])) >= -int(np.round(whichhascapacity[0][0])):
                    outlist[ whichexceed[0][1] ] += int(np.round(whichhascapacity[0][0]))
                    outlist[ whichhascapacity[0][1] ] -= int(np.round(whichhascapacity[0][0]))
                else:
                    outlist[ whichexceed[0][1] ]  -= int(np.round(whichexceed[0][0]))
                    outlist[ whichhascapacity[0][1] ] += int(np.round(whichexceed[0][0]))
    else:
        
        whichexceed = []
        sumexcess = 0
        whichhascapacity = []
        sumcapacity = 0

        # NOTE WE ADD FUDGE FACTORS TO THE EXCESS; if we don't do this then when we sort, we'll always have lower indices appear first, and we want to randomize those
        for i in range(N_arc):
            excess = np.max([0, -outlist[i] - CUTOFF])
            if CUTOFFVOLUME != 0:
                excess = np.min([excess, CUTOFFVOLUME])
            if excess > 0:
                whichexceed.append((excess + np.random.normal()/100., i))
                sumexcess += excess

            capacity = np.max([0, outlist[i]])
            if capacity > 0:
                whichhascapacity.append((capacity + np.random.normal()/100.0, i))
                sumcapacity += capacity

        whichexceed.sort()
        whichexceed.reverse() 
        whichhascapacity.sort()

        # let's start with the simplest choice (more or less) of just annihilating the biggest excsss with the biggest capacity
        if rn.random() < FAC:                    
            if sumcapacity > 0 and sumexcess > 0:
                if int(np.round(whichexceed[0][0])) >= int(np.round(whichhascapacity[0][0])):
        
                    outlist[ whichexceed[0][1]] += int(np.round(whichhascapacity[0][0]))
                    outlist[ whichhascapacity[0][1]] -= int(np.round(whichhascapacity[0][0]))
                else:
                    
                    outlist[ whichexceed[0][1]] += int(np.round(whichexceed[0][0]))
                    outlist[ whichhascapacity[0][1]] -= int(np.round(whichexceed[0][0]))
    #outspan = max(outlist) - min(outlist)

    return outlist


def findintercept(x,yobs,m):
    m2forced = np.round(2*m)
    mchosen = m2forced/2.0
    return( np.mean(np.array(yobs)) - mchosen * np.mean(np.array(x)) )



def plottingstuff(opts, tarr, y):


    #   0          1        2       3             4
    #(tHisTArr, absTotArr, l2Arr, sumgridArr, sumgrid2Arr)

    outtup = (tarr, y)

    whichoutput = 1
    whenstart = 0

    plt.plot(tarr, outtup[whichoutput])  # used to be 1 instead of -1
    plt.show()


    plt.plot(np.log(1.0+np.array(outtup[0])), np.log(np.array(outtup[whichoutput])))  # used to be 1 instead of -1
    plt.show()


    #plt.plot(outtup[0], outtup[1]) # used to be 2 instead of -1
    #plt.plot(outtup[0], np.sqrt(np.array(outtup[whichoutput]))) # used to be 2 instead of -1
    #plt.show()


    a = np.log(1.0+np.array(outtup[0][whenstart:])) 
    b = np.log(np.array(outtup[whichoutput][whenstart:])) # used to be 1 instead of -1

    plt.plot(a, b)


    sig = np.std(np.array(outtup[whichoutput][1:])-np.array(outtup[whichoutput][:-1]))
    mu = np.mean(np.array(outtup[whichoutput][1:])-np.array(outtup[whichoutput][:-1]))

    acc = np.mean(np.array(outtup[whichoutput][2:])-2*np.array(outtup[whichoutput][1:-1])+np.array(outtup[whichoutput][:-2]))

    

    thisintercept = findintercept(tarr, y)
    print("OUTPUT", sys.argv)
    print(linregress(a, b), thisintercept, np.exp(thisintercept))


def GetDestination(xcoord, idir, lngth):
    idim = idir // 2
    newcoord = list(xcoord)
    oldcoorddim = newcoord[idim]
    if idir % 2 == 0:
        # go "up"
        newcoord[idim] = (oldcoorddim + 1) % lngth
    else:
        # go "dn"
        newcoord[idim] = (oldcoorddim - 1) if oldcoorddim > 0 else lngth - 1
    return newcoord


def ExportDimGraph(ggraph, bCont=False): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.
    """

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    if bCont:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim]))
    else:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])

            if bCont:
                retarr[thiscoord] = i.ContAmplitude[idir]
            else:
                retarr[thiscoord] = i.Amplitude[idir] 
    return retarr


def ExportDimGraphCtl(ggraph, bCont=False): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.
    """

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    if bCont:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim]))
    else:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])

            if bCont:
                retarr[thiscoord] = i.ContAmplitude[idir]
            else:
                retarr[thiscoord] = i.AmplitudeCtl[idir] 
    return retarr





def ExportDimGraphx(ggraph, bCont=False): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.

    This version corrects for the fact that the Z-tokens will be offset so that
    the whorl contribution at any node is zero, as it should be.
    """

    def GetDestination(xcoord, idir, lngth):
        idim = idir // 2
        newcoord = list(xcoord)
        oldcoorddim = newcoord[idim]
        if idir % 2 == 0:
            # go "up"
            newcoord[idim] = (oldcoorddim + 1) % lngth
        else:
            # go "dn"
            newcoord[idim] = (oldcoorddim - 1) if oldcoorddim > 0 else lngth - 1
        return newcoord

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    if bCont:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim]))
    else:
        retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])

            if bCont:
                retarr[thiscoord] = i.ContAmplitude[idir]
            else:
                retarr[thiscoord] = i.Amplitude[idir] - i.ZAmplitude[idir]
    return retarr


def ZExportDimGraph(ggraph): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.
    """

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])
            if ggraph.bNeedZTokens:
                retarr[thiscoord] = i.ZAmplitude[idir]

    return retarr


def NonZeroSection(arrx):

    start = len(arrx)
    for i in range(len(arrx)-1, -1, -1):
        if arrx[i] > 0:
            start = i + 1
        else:
            break   
    return start

def Excess(inlist, M=0):

    N_arc = len(inlist)
    excessarr = np.zeros(N_arc).astype("int")

    for ii, iin in enumerate(inlist):
        excessarr[ii] = np.max([0, iin - M]) if iin >= 0 else np.min([0, iin + M])
    
    
    unmatched = FillIn(excessarr)
    matched = excessarr - unmatched

    return matched, unmatched


def SplitMatchedUnmatched(inlist, M=0):
    
    unmatched = FillIn(inlist)
    matched = np.array(inlist).astype("int") - unmatched

    return matched, unmatched

def ScrubWhorls(inlist, inlit_br):
    """
    If there is no copacity, turn around...
    """
    return None


def WipeWhorlsAtSingleTime(ggraph, M=0):
    """
    Scrub the graph at a single time to get rid of whorls
    """

    FillFn = FillInFromBase # try different ones 

    arngshuff = np.arange(ggraph.NNode)

    bShuffleAtEachStep = True
    if bShuffleAtEachStep:
        rn.shuffle(arngshuff)


    previ = -1
    for i in arngshuff:
        inlist = []

        inlist_ztoken = []
        inlist_ctl = []



        for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
            whichnbr = ggraph.NodeVec[i].FromNbr[iinbr]   

            inlist.append(ggraph.NodeVec[inbr].Amplitude[whichnbr])
            if ggraph.bNeedZTokens:
                inlist_ztoken.append(ggraph.NodeVec[inbr].ZAmplitude[whichnbr]) 
                inlist_ctl.append(ggraph.NodeVec[inbr].AmplitudeCtl[whichnbr])
        previ = copy(i)

        if  bContinuous:
            inlist = []
            for iinbr, inbr in enumerate(ggraph.NodeVec[i].Neighbors):
                whichnbr = ggraph.NodeVec[i].FromNbr[iinbr]            
                inlist.append(ggraph.NodeVec[inbr].ContAmplitude[whichnbr])
            floatingoutamp = FloatingPtWave(inlist)
            ggraph.NodeVec[i].ContScratch = floatingoutamp
        else:

            ggraph.NodeVec[i].ZScratchIn = inlist_ztoken
            outamp_ctl = ModPool(inlist_ctl)  
            #comparison = outamp_ctl
            #incomparison = inlist_ctl

            #if  np.sum(np.abs(inlist_ztoken)) != 0:
            #    print("ay")
            #    import pdb; pdb.set_trace()
            outamp, outampz = ScrubWhorls(inlist, inlist_ztoken, CoreFn)    


            if False:


                if i == 560 and np.sum(outamp_ctl) == 1:
                    print("560,1")
                    import pdb; pdb.set_trace()

                if np.max(outamp) * np.min(outamp) < 0:
                    print("outneg")
                    import pdb; pdb.set_trace()

                if i == 495 and np.sum(np.abs(inlist_ztoken)) != 0 :
                    print("495,inznotzero")
                    import pdb; pdb.set_trace()

                if False and np.sum(outamp + outampz) != np.sum(outamp_ctl):
                    print("here's 527")
                    import pdb; pdb.set_trace()

                if False and np.sum(outamp ) != np.sum(outamp_ctl):
                    print("mismatch")
                    import pdb; pdb.set_trace()
                if False and np.sum(outamp) != np.sum(outamp_ctl):
                    print("mismatch")
                    import pdb; pdb.set_trace()
              



            ggraph.NodeVec[i].PrevAmplitude = copy(ggraph.NodeVec[i].Amplitude)
            ggraph.NodeVec[i].Scratch = outamp
            
            if ggraph.bNeedZTokens:
                ggraph.NodeVec[i].ZScratch = outampz
                ggraph.NodeVec[i].ScratchCtl = outamp_ctl
        
        
    if  bContinuous:
        for i in arngshuff:
            ggraph.NodeVec[i].ContAmplitude = ggraph.NodeVec[i].ContScratch
    else:
        for i in arngshuff:
            ggraph.NodeVec[i].Amplitude = ggraph.NodeVec[i].Scratch
        
            
            if ggraph.bNeedZTokens:                

                # Note that when exporting to an array (incase of a D-dimensional array setup) 
                # there needs to be an offsetting Z-amp update at the DESTINATION of the outgoing arc
                ggraph.NodeVec[i].ZAmplitude = ggraph.NodeVec[i].ZScratch
                ggraph.NodeVec[i].AmplitudeCtl = ggraph.NodeVec[i].ScratchCtl
                #ggraph.NodeVec[i].ZAmplitudeIn = ggraph.NodeVec[i].ZScratchIn


def CtlExportDimGraph(ggraph): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.
    """

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])
            retarr[thiscoord] = i.AmplitudeCtl[idir]
            #if i.ZAmplitude[idir] != 0:
            #    print("nana")
            #    import pdb; pdb.set_trace()
    return retarr


#deprecated
def ZInExportDimGraph(ggraph): 
    """
    Use this routine to dump the amplitudes of a D-dimensional graph onto
    a D+1-dimensional array (with the last dimension being the 2D directions),
    for easier plotting and analysis.
    """

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))
    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim])).astype("int")

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        for idir in range(2*ggraph.NDim):
            thiscoord = tuple(list(i.Coords) + [idir])
            #retarr[thiscoord] = i.ZAmplitudeIn[idir]
    return retarr



def ExportDimGraphNodeNumber(ggraph): 

    """
    ONLY FOR DEBUGGING: Use this routine to print the coordinates.

    """



    def FindNbrCoords(xcoords, length):
        retlist = []
        for i in range(len(xcoords)):
            upcoord = copy(xcoords)
            dncoord = copy(xcoords)
            upcomp = (xcoords[i] + 1) % length
            dncomp = length - 1 if xcoords[i] == 0 else xcoords[i] - 1
            upcoord[i] = upcomp
            dncoord[i] = dncomp
            retlist.extend([upcoord, dncoord])
        return retlist

    # return D-dimensional array of amplitudes for easier analysis
    if ggraph.NNode == 0 or ggraph.NDim == 0:
        return np.array([])
    lngth = int(np.round(ggraph.NNode ** (1.0/ggraph.NDim)))


    shapetuple = tuple(np.ones((ggraph.NDim,)).astype("int") * lngth)
    
    
    #retarr = np.zeros(tuple(list(shapetuple) + [2 * ggraph.NDim]))
    retarr = np.zeros(shapetuple)

    for ii, i in enumerate(ggraph.NodeVec):
        coords = i.Coords
        xcoords = coords
        xnbrcoords = FindNbrCoords(xcoords, lngth)
        retnbrlist = [0 for j in xnbrcoords]
        
        thiscoord = tuple(list(coords))
        retarr[thiscoord] = int(ii)
    return retarr    


def LocateNonZeroZ(ggraph):
    # For debugging only -- print out the coordinates for a given point
    for inod in ggraph.NodeVec:
        for iinbr, inbr in enumerate(inod.Neighbors):
            if inod.ZAmplitude[iinbr] != 0:
                print("NZ irun", ggraph.irun, "t", ggraph.t, "x", inod.Coords, iinbr, inod.ZAmplitude[iinbr])




def NeighborCoords(ggraph, coordstup):
    # For debugging only -- print out the coordinates for a given point

    coordstup = tuple(coordstup)
    for inod in ggraph.NodeVec:
        import pdb; pdb.set_trace()
        if coordstup == tuple(inod.Coords):
            for iinbr, inbr in enumerate(inod.Neighbors):
                print(iinbr, ggraph.NodeVec[inbr].Coords)
            return


def do3dgraphof2dmatrix(ggraph, idir=0):
    thisarr = ExportDimGraph(ggraph)
    # first do x*x

    # https://www.reddit.com/r/learnpython/comments/16kaxst/how_do_i_plot_a_2d_surface_3d_plot_using/
    xdiff = 1.0
    x = np.arange(0, thisarr.shape[0], xdiff)
    y = np.arange(0, thisarr.shape[1], xdiff)

    x = x - np.mean(x)
    y = y - np.mean(y)

    import pdb; pdb.set_trace()
    xx,yy = np.meshgrid(x,y)


    #zz = thisarr[:, :, idir]  # Or any other definition.

    # squared sum of the function along the arcs (note the sum itself is always identically zero, by)
    zz = np.sum(thisarr[:, :, :] * thisarr[:, :, :])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def CreateZerosFromGraph(ggraph):
    myret = []
    for inode in ggraph.NodeVec:
        thislist = [0 for inbr in inode.Neighbors]
        myret.append(thislist)
    return myret

def AddByGraphNeighbors(ListOfListsA, ListOfListsB):
    myret = []
    for i in range(len(ListOfListsA)):
        thislist = [(ListOfListsA[j] + ListOfListsB[j]) for j in range(len(ListOfListsA[i])) ]
        myret.append(thislist)
    return myret




def dosomegraphsatcenter(ggraph,ngraph,bXdir=True):
    thisarr = ExportDimGraph(ggraph)
    # first do x*x


    if bXdir:
        centr = thisarr.shape[1]//2-1
        chunk = thisarr[:,(centr-ngraph//2):(centr+ngraph//2), 0]
    else:
        centr = thisarr.shape[0]//2-1
        chunk = thisarr[(centr-ngraph//2):(centr+ngraph//2), :, 0].T

    plt.plot(np.arange(thisarr.shape[1]), chunk)
    plt.show()
        

    b3D_meshgraph = False

    if b3D_meshgraph:
    # https://www.reddit.com/r/learnpython/comments/16kaxst/how_do_i_plot_a_2d_surface_3d_plot_using/
        xdiff = 1.0
        x = np.arange(0, thisarr.shape[0], xdiff)
        y = np.arange(0, thisarr.shape[1], xdiff)

        xx,yy = np.meshgrid(x,y)
        zz = thisarr[:, :, 0]  # Or any other definition.


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, yy, zz)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    elif False:
        a = thisarr[:,:,0]
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("slice 0")
        plt.show()

        a = thisarr[:,:,1]
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("slice 1")
        plt.show()

        a = thisarr[:,:,2]
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("slice 2")
        plt.show()

        a = thisarr[:,:,3]
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("slice 3")
        plt.show()


        print("inspectaway")
        import pdb; pdb.set_trace()



def Main():



    opts, args = ReadParams()

    bListInitializerFunctions = False

    BoundaryConditions = opts.initialconditions



    if len(sys.argv) > 1:
        if sys.argv[1] == '--initlist' and len(sys.argv) == 2:
            bListInitializerFunctions = True



    fnlist = []

    if opts.dimension == 0:
        desctxt = "BoundaryConditions: single discrete Dirac delta at the center of the lattice"
    else:
        desctxt = "BoundaryConditions: single discrete Dirac delta functions at an edge connecting node 0 to a nearest neighbor"


    maxtxtlen = 60

    fnlist.append((InitializeDiracDelta, 'InitializeDiracDelta', desctxt))

    fnlist.extend([(InitializeDiracDelta2, 'InitializeDiracDelta2', "BoundaryConditions: same as InitializeDiracDelta, but scaled by 2. "),
        (InitializeDiracDelta4, 'InitializeDiracDelta4', "BoundaryConditions: same as InitializeDiracDelta, but scaled by 4 (therefore potentially problematic in terms of PEP in the 2-D case, and therefore requiring displacement)." ),
        (InitializeIncoming3pos1neg, 'InitializeIncoming3pos1neg', "BoundaryConditions: discrete Dirac delta on edges converging at a node at the center of the lattice; one of the Dirac delta functions will be negated, and in the 2-dimensional case, the subsequent step will correspond to a hodotic solution (see text) of amplitude 2 and will require Z-token shifting. "),
        (Initialize8off, 'Initialize8off', "BoundaryConditions: a configuration of hodotic solutions arranged to force an amplitude of 4 in two steps. "),
        (Initialize8offA, 'Initialize8offA', "BoundaryConditions: much like Initialize8off, but all the initial hodotic solutions are of the same sign."),
        (InitializeNaiveRegEmission, 'InitializeNaiveRegEmission', "BoundaryConditions: a particle/antiparticle pair." ),
        (InitializeNaiveRegEmissionWWhorl, 'InitializeNaiveRegEmissionWWhorl', "BoundaryConditions: particle/antiparticle, along with an offsetting whorl (see code for details)." ),
        (Initialize1Zwhorl, 'Initialize1Zwhorl', "BoundaryConditions: a whorl-like arrangement of tokens (with no corresponding regular particles) that will vanish after one step." ),
        (Initialize2CollideAngle, 'Initialize2CollideAngle', "BoundaryConditions: visually, a kind of scattering situation with two hodotic solutions colliding, so to speak, at an an angle (opposite sign)." ),
        (Initialize2CollideAngleSameSign, 'Initialize2CollideAngleSameSign', "BoundaryConditions: visually, a kind of scattering situation with two hodotic solutions colliding, so to speak, at an an angle (same sign)." ),
        (Initialize2CollideHeadOn, 'Initialize2CollideHeadOn', "BoundaryConditions: visually, a kind of scattering situation with two hodotic solutions colliding, so to speak, from opposite directions (opposite sign)." ),
        (Initialize2CollideHeadOnSameSign, 'Initialize2CollideHeadOnSameSign', "BoundaryConditions: visually, a kind of scattering situation with two hodotic solutions colliding, so to speak, from opposite directions (same sign)." ),
        (Initialize1whorlA, 'Initialize1whorlA', "BoundaryConditions: simple whorl emission. Statistically identical to emitting no particles at all (but requiring many more steps to converge for higher number of steps, than with ModPool dynamics)." ),
        (Initialize1shift, 'Initialize1shift', "BoundaryConditions: displacing a particle by way of single-shifting." ),
        (Initialize2offC, 'Initialize2offC', "BoundaryConditions: simple whorl emission." ),
        (Initialize2offD, 'Initialize2offD', "BoundaryConditions: another simple whorl emission." ),
        (Initialize1cluster, 'Initialize1cluster', "BoundaryConditions: simple cluster arrangement with one having opposite sign (designed for 2-D rectangular case)." ),
        (Initialize1shift, 'Initialize1shift', "BoundaryConditions: simple whorl emission for the purpose of single-shifting a particle from one arc to another." ),
        (Initialize2shift, 'Initialize2shift', "BoundaryConditions: simple whorl emission for the purpose of doublee-shifting a particle/antparticle pair away." ),
        (Initialize2off, 'Initialize2off', "BoundaryConditions: particle/antiparticle emission (as with naive regularization)." ),
        (Initialize1plaquette, 'Initialize1plaquette', "BoundaryConditions: whorl-like arrangement of regular particles (that in ModPool dynamics can persist indefinitely)." ),
        (Initialize1plaquetteZ, 'Initialize1plaquetteZ', "BoundaryConditions: whorl-like arrangement of tokens that vanishes in the subsequent step" )
    ])


    if bListInitializerFunctions:

        longtxt = "The following is a list of numerous simple boundary conditions \
(the first word of any paragraph is to be used with the --boundaryconditions parameter in any command line. \
(Run the full tutorial to see working examples of command lines and, if desired, swap out the listed initializer \
functions with one of the ones here). Note that the routines here are for  \
D-dimensional rectangular graphs (with the two dimensional case being the most practical one; detailed evolution \
may be observed for many of these routines with the assistance of the --print option, as noted in the full tutorial):"
        print(" ")
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        for ifn, istr, idesc in fnlist:
            print(textwrap.fill((istr + " " + idesc.replace('BoundaryConditions:', ':'))))
            print(" ")
        sys.exit()

    bFoundAmongMiscInitRoutines = False
    for ifn, istr, desctxt in fnlist:
        if BoundaryConditions == istr:
            InitializerFn = ifn
            bFoundAmongMiscInitRoutines = True
            break


    if not(bFoundAmongMiscInitRoutines) and opts.dimension > 0:
        print("Unrecognized boundary condition -- check the available ones and find the correct name, or add your own routine.")
        sys.exit()


    print("\n" + desctxt + "\n")






















    InitializerFn = InitializeDiracDelta


    bDiracDeltaInitialize = True

    if opts.dimension == 0 and opts.loadpicklefile == "" and opts.loadadjacencyfile == "":
        bDiracDeltaInitialize = False

    if opts.dimension > 0 and (opts.length % 2 == 1):
        print("ERROR: ")



    if opts.tutorial and len(sys.argv) > 2:
        xtraswitches = []
        for i in sys.argv[1:]:
            if i != '--tutorial':
                xtraswitches.append(i)

        print("ERROR: if you wish to use the --tutorial switch, it must be the only switch in the command line. Either remove the --tutorial or remove \n   " + ' '.join(xtraswitches) + "\n and rerun. ")
        sys.exit()



    if opts.tutorial:
        print(" ")
        longtxt = "Before proceeding to the tutorial, would you like to try out some sample analyses just to get a quick overview of how the code is used?"
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")
        print("Answer 'yes' or 'no'")
        Input0 = input().strip()

        bDeepDive = False
        if len(Input0) > 0:
            if Input0[0] == 'y' or Input0[0] == 'Y':
                bDeepDive = True
        
        if bDeepDive:
            longtxt = "In that case, copy and paste either of the following python commands into the terminal (requires basic familiarity with running Python scripts on your computer). The first one constructs a graph \
consisting of a 2-dimensional 8x8 lattice (as dictacted by the --dim and --length parameters) and performs \
a MonteCarlo computation of 1000 runs (as indicated by the --runs parameter), with each run consisting of 3 steps (as dictated by the --steps parameter) \
of BHP propagation that satisfiesa Pauli Exclusion Principle (as dictated by the \"--dynamics fermi\" selection), always starting each run with a Dirac delta function (as indicated by the \"--boundaryconditions InitializeDiracDelta\" selection). \
At every node of the graph, the incoming/outgoing net particle counts at the end of each run are averaged and then the squared differences of these averages from the corresponding results of a 3-step run of the continuous case is summed over the graph. Additionally, \
as a control case, the same number of runs/steps are repeated with a system of particles obeying ModPool dynamics, so as to allow for a comparison between ModPool and whatever dynamics the user has chosen, and both squared-error subs are reported to show \
how the vales converge to the correct continuous values as the number of MonteCarlo runs increases. \
If you  wish to see a list of the installed boundary conditions available for rectangular lattices, run: python " + __file__ + " --initlist "
            print(" ")
            print(" ")
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            print("python " + __file__ + " --dim 2 --length 8  --steps 3  --dynamics fermi  --seed 137 --runs 1000  --boundaryconditions InitializeDiracDelta")
            print(" ")
            longtxt = "The --seed parameter is arbitrary. \
If the --dim parametr is 2, then after every thousand runs or so, there will also be a printout of the matrix representations of both the BHP-fermi and ModPool MonteCarlo distributions."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            longtxt = "Or else, in order to run a long-term-particle-growth analysis (for which the --runs parameter is set to 1, and the --steps parameter is set to a high number), run the following command. \
If the relevant dynamics involves Z-tokens, be aware that for shorter runs (sometimes that means even runs of less than several billion), the growth rate will deviate substantially from 0.5."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            print("python " + __file__ + "  --dim 2 --length 8  --steps 5_000  --dynamics fermi  --seed 137 --runs 1  --boundaryconditions InitializeDiracDelta")
            print(" ")
            longtxt = "By varying the graph sizes (in general, if the --dim switch is invoked, the graph in question is a D-dimensional rectangular lattice with D**L nodes, where the D and L parameters are specified by --dim and --length, and where you should ensure that D**L is reasonably low to start with) and also the run/step parameters, and trying different boundary conditions (a listing can be seen by running \"python " + __file__ + " --initlist \"), then anyone with some familiarity of Python can gain sufficient familiarity with the code to be able to modify it to suit other areas of interest."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            longtxt = "If all that is too much and too fast to take in at once, then consider this as just a preview of the rest of this tutorial, and answer \"no\" the next time you run it."
            print(" ")
            print(textwrap.fill(longtxt, width=maxtxtlen))


        if bDeepDive:
            sys.exit()

        print(" ")
        longtxt = "This program runs MonteCarlo simulations of various Brownian-Huygens propagation models, and \
can be used to generate all the data in the accompanying paper. There are two primary modes of operation, \
depending on the value of the --runs (the number of MonteCarlo runs) and --steps (the number of steps in each run) \
parameters (call them NRuns and NSteps):"
    
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")
        longtxt = "1. Convergence studies that demonstrate that \
the specified dynamics of particles (and if needed, for Z-tokens) yield distributions whose expected values converge to the results of the continous (discrete) wave solution (and \
which furthemore compare the rate of convergence with that of the ModPool BHP dynamics, which is taken as a control case). For this mode of operation, \
the program will first calculate for NSteps steps the continuously-valued wave equation, starting with the specified lattice and boundary conditions\
(note that for generalized random graphs, the resultant values may have \
little or no relationship to the classical wave equation as obtained by way of rectangular lattices). \
Then, the program will calculate NRuns MonteCarlo simulations, each of them likewise starting with the specified boundary conditions, \
and continuing for NSteps steps. The particle counts after each run will be calculated at each node of the graph, and then the squared \
difference of the observed average and the continous-case will be summed over all nodes so to obtain an overall error term. As \
the number of runs increases, the error term will eventually decrease (in fluctuating fashion, since it, too, is a random variable)."
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")



        longtxt = "2. Growth analyses that determine how the absolute (i.e. positive plus negative) particle count in the graph (and where appropriate, positive plus negative z-token count) increase with time (i.e. with the number of steps) for any single run. \
For this mode of operation, NRuns is set to 1 and NSteps to a very large number (though frequently, a few thousand will be enough to give a reasonable estimate of the long-term behavior), and \
then periodically the log of the (relevant) particle count vs. the log of time is regressed to calculate the growth rate."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "The enclosed code also provides several ways of creating non-standard graphs (as well as rectangular lattices in numerous dimensions). \
If you wish to learn about these different options for generating different graphs, or loading your own that were created elsewhere, answer 'yes'; if instead you would prefer to skip that and \
focus on D-dimensional rectangular latices, answer 'no'. (Keep in mind that \
non-standard graphs need not have an integer dimension even in \
the large-node limit, and their random walk particle distributions and BHP particule distributions need not have any connection to the heat equation or wave equation, respectively.)"
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")
        print("Answer 'yes' or 'no'")
        Input0 = input().strip()

        bSkipToDDimensionalGraphs = True
        if len(Input0) > 0:
            if Input0[0] == 'y' or Input0[0] == 'Y':
                bSkipToDDimensionalGraphs = False
        
        
        if not(bSkipToDDimensionalGraphs):
            
            
            longtxt = "A generic graph, including any particle counts, or amplitudes, on its edges, are saved and loaded by way of simple input files.\
To see how this works in practice, consider a graph with 64 nodes arranged as a hexagonal \
(two-dimesnional) lattice."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print("\n\n    |     |     |     |     |     |     |     |")
            print("   56    57    58    59    60    61    62    63")
            print("     \  /  \  /  \  /  \  /  \  /  \  /  \  /  \  /")
            print("      48    49    50    51    52    53    54    55")
            print("       |     |     |     |     |     |     |     |")
            print("      40    41    42    43    44    45    46    47")
            print("     /  \  /  \  /  \  /  \  /  \  /  \  /  \  /  \ ")  
            print("   32    33    34    35    36    37    38    39")
            print("    |     |     |     |     |     |     |     |")
            print("   24    25    26    27    28    29    30    31")
            print("     \  /  \  /  \  /  \  /  \  /  \  /  \  /  \  /")   
            print("      16    17    18    19    20    21    22    23")
            print("       |     |     |     |     |     |     |     |")
            print("       8     9    10    11    12    13    14    15")
            print("     /  \  /  \  /  \  /  \  /  \  /  \  /  \  /  \ ")  
            print("    0     1     2     3     4     5     6     7   \n\n")
            print(" ")
            longtxt = "Assume the edges at the periphery of the graph are connected toroidally, so that \
node 0 is connected not just to 8 but also nodes 56 and 15, and so on."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            longtxt = "We can specify this graph with an input file of 64 lines, \
one for every node of the graph, with each line consisting of N \
(comma-delimited) unordered strings corresponding to the N neighbors of that node. Each of these strings \
is itself a (period-delimited) string of one to 3 integers: the initial \
sub-substring specifies a neighbor node index, then the amplitude on the edge \
that connects the designated node pertaining to that line to that neighbor, \
followed by the z-token amplitude; if the substring has only one element, \
then both the amplitude and z-token amplitude \
for that edge are zero, and if the substring has only two elements, then the \
z-token amplitude is zero. So. for the above graph, assuming the amplitude on the edge \
connecting node 0 and node 8 is 1 (with all other amplitudes and Z-amplitudes on the \
graph being zero, the initial line of the input file (representing information about \
node 0) would be: '8.1.0,15,56'\n"
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            longtxt = "(Note again that any ignored/omitted element of the 3-tuple substrings \
pertaining to a given node are 0. The subsequent line (rereffing to node 1) \
would be 8,9,57 (or, equivalently, 8;0;0,9;0;0,57;0;0). The complete file for this hexagonal graph is \
available in the code directory as adjacencylist_hexagonal64.csv, along with \
the adjacency matrix of the graph connectiondiagram_hexagonal64_noamplitudeinfo.csv \
(note the latter contains no amplitude information). A corresponding triangular lattice (also with hexagonal symmetry, but where each node is \
the central point of a hexagon defined by six nearest neighbors, and is, up to a scaling transformation, the same as a 2-D \
rectangular array in which each node has not only the usual 4 up/down/left/right, but also two additional diagonal neighbors along the 45 degree angle from the horizontal) \
is contained in adjacencylist_triangular64.csv. Note the latter graph is not bipartite."
            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")
            longtxt = "Alternatively, you can choose as your graph a D-dimensional rectangular lattice of length --length in each \
dimension, subject to the restriction that D**length < maxnodes; maxnodes is currently set to 500,000 and may be adjusted (or else, the size restriction check may be removed altogether). Type any key to proceed to the case of D-dimensional lattices..."  

            print(textwrap.fill(longtxt, width=maxtxtlen))
            print(" ")




        longtxt = "Assuming a D-dimensional lattice of length L in each dimension, as specified by the --dimension and --length switches \
(so that the number of nodes is D**L), there are several initializations available. Here are a few (see the code for some more)."
        print(textwrap.fill(longtxt, width=maxtxtlen))

        print(" ")


        longtxt = "    InitializeDiracDelta -- initialize with a discrete Dirac delta function in the \"middle\" of the space. While it is a useful starting point, \
and is the proposed initialization for the graphs that are not D-dimensional lattices, they are not particularly useful in testing the convergence of any dynamics involving Z-tokens. \
The reason for this is that there are relatively few violations of the Pauli Exclusion principle even when using standard ModPool dynamics, and the convergence rates of both models will \
be fairly similar. A much better choice is a set of boundary \
conditions where the violations of the PEP are frequent and inevitable, and in those cases, it will be seen that the convergence of the Z-token dynamics will be significantly slower, as it should be. \
\n\More useful in that regard are InitializeDiracDelta2 and  InitializeDiracDelta4 (the same conditions multipled by 2 and 4, with the latter being the largest amplitude an edge on a two-dimensional \
rectangular lattice can hold without leading to violations of the PEP)  with the --steps parameter set to 3 or higher."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print("Sample script (in the list that follows, substitute the --boundaryconditions parameter with any of the other choices) listed below: ")
        print("    python " + __file__ + " --dim 2 --length 8 --steps 3 --dynamics bosevary --seed 579202 --runs 1_000 --boundaryconditions InitializeDiracDelta")
        print(" ")

        

        longtxt = "    InitializeWhorl -- initialize with the emission of a whorl -- i.e. a particle-antiparticle pair emanating from a single node along with a token/antitoken pair in the same node. From a statistical perspective, this starting state is equivalent to having no particles at all. See how it converges to that no-particle state shen --steps is set to 3 or more."


        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        

        longtxt = "    InitializeNaiveRegEmission -- initialize with a large naive regularization emission (i.e. two scaled hodotic solutions emanating from the same point). "


        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")



        longtxt = "    InitializeNaiveRegEmissionWWhorl -- same as InitializeNaiveRegEmission, but with a whorl emission"

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "    InitializeNaiveRegEmissionWTokens -- similar to InitializeNaiveRegEmissionWWhorl, but with the particles of the whorl and the naive-regularization cancel out, leaving only tokens. \
This demonstrates how naive regularization may be effected through Z-token/antitoken emissions alone (so that any PEP restrictions are obeyed, since no particle/ emissions are involved.) "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "    InitializeIncoming3pos1neg -- in two dimensions, this forces an amplitude of 2 (i.e. a violation of the PEP when effected by way of ModPool, but which can be displaced away by the use of whorls) in the subsequent steps. Set the --length parameter to 8 and the --steps parameter to 3. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "    Initialize8off -- in two dimensions, this forces an amplitude of 4 (i.e. a violation of the PEP when effected by way of ModPool, but which can be displaced away by the use of whorls) in the subsequent steps. Set the --length parameter to 8 and the --steps parameter to 3. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "    Initialize8offA -- a minor variation of Initialize8off that leads to an amplitude of 4 at a single node, and which can also be displaced away by the use of whorls) in the subsequent steps. Set the --length parameter to 8 and the --steps parameter to 3. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        

        longtxt = "    Initialize2CollideAngle -- Two hodotic solutions (of opposite sign) converging/colliding into the same node at a right angle. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "    Initialize2CollideAngleSameSign -- Two hodotic solutions (with same sign) converging/colliding into the same node  at a right angle. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")



        longtxt = "    Initialize2CollideHeadOn -- Two hodotic solutions (of opposite sign) converging/colliding into the same node with opposite direction. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        longtxt = "    Initialize2CollideHeadOnSameSign -- Two hodotic solutions (with same sign) converging/colliding into the same with opposite direction. "

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "In the special case of two-dimensional rectangular lattices, the --print option provides a 3-fold printout \
of amplitudes on the graph edges for particles, z-tokens, and also the  ModPool (i.e. the  \"control\") cases. (The incoming/outgoing sums of these edge flows at any node are also printed.) \
The feature was particularly useful in debugging and modifying the dynamics. \
It is simplest to use on lattices whose --length parameter is 4 but can \
easily be adapted to 4x4 windows of larger graphs (or 2-dimensional 4x4 \
slices of higher dimensional spaces). \
So if you are curious about how particle counts change from step to step for a given set of dynamics, set the --dim and --length  parameters to 2 and 4, \
respectively, and see how a given set of boundary conditions evolves step by step. Be aware that the x-y orientation of the printouts is flipped relative \
to the matrix displays (which are obtained via the routine ExportDimGraph()."
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "Another useful routine for figuring out which node is situated where on the graph, try ExportDimGraphNodeNumber() which prints out the node indices at each location. \
slices of higher dimensional spaces). So if you are curious about how particle counts change from step to step for a given set of dynamics, set the --dim and --length  parameters to 2 and 4, respectively, and see how a given set of boundary conditions evolves step by step."
        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "The user is encouraged to construct other boundary conditions of interest. As a start, it is strongly suggested to work with boundary conditions for which the Z-token count is \
everywhere zero (or else where a whorl has been added as with InitializeNaiveRegEmissionWWhorl). Also, restrict the nonzero amplitudes to nodes whose coordinates X have the property that the sum of \
their D components is even. \n\nAn additional word of warning for those who want to modify the dynamics: It is very easy to construct models \
using Z-tokens where both the absolute particle count and the token count are bounded. Based on the author's experience, any such model probably has some error, \
so that its expected distribution values do not align with the wave equation. The error \
will not always be easily observable, especially if you start with a single Dirac delta function as the boundary condition, but for long enough simulation intervals, and high enough  MounteCarlo runs, \
the  failure to converge became evident. The error will also be much easier to spot with boundary conditions corresponding to two or more hodotic \
solutions colliding into one another and then testing MonteCarlo distributions for simulation intervals t > 3. \
Removing those errors inevitably led to a sqrt(T) rise in the token count. That can of course be removed by naive regularization, but that, too, is \
a corruption of the wave equation, though one that can be made arbitrarily minimal by setting the M threshold to be a large number."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "You can save (as a pickle file) any graph G at any time using the SaveGraph(G, dumppklname) or SaveGraphWOpt(ggraph, dumppklname) \
routines, and then if you run a script with dumppklname as the --loadfrompkl parameter, the initial conditions will be the graph that was saved."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        longtxt = "Here are a few other scripts that may be of particular interest in that they demonstrate the difference (in terms of rates of convergence) between whorl-regularized systems and plain ModPool (especially for longer and longer values of the --steps parameter):"

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")



        longtxt = "In this one, with two hodotic solutions \"collide\", so to speak."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")


        print("    python " + __file__ + " --dim 2 --length 4 --steps 3 --dynamics bosevary --seed 579202 --runs 1_000 --boundaryconditions InitializeNaiveRegEmission")

        longtxt = "Please report any bugs or other suggestions to the author at the email address given in the paper."

        print(textwrap.fill(longtxt, width=maxtxtlen))
        print(" ")

        sys.exit()
       

    argdict = {}

    argdict["Reduction"] = 0

    if opts.steps == 0:
        print("ERROR: --steps parameter must be > 0 (or else just modify the code a little.)")
        sys.exit()


    if bDiracDeltaInitialize:
        InitializerFn = InitializeDiracDelta
        if opts.dimension == 0:
            argdict["initializationdescription"] = "BoundaryConditions: single discrete Dirac delta at the center of the lattice"
        else:
            argdict["initializationdescription"] = "BoundaryConditions: single discrete Dirac delta at node 0"












    bHeatEq = False
    # the following two options calculate the heat equation (either as particles executing random walks, or else the modpool version thereof)
    if opts.dynamics in ['pureheat', 'randomwalk']:
        argdict['InOutFunction'] = PureRandomWalkHeat
        argdict['CoreFn'] = PureRandomWalkHeat
        bHeatEq = True
    elif opts.dynamics in ['heat', 'modpoolheat']:
        argdict['InOutFunction'] = ModPoolJustBrownian
        argdict['CoreFn'] = ModPoolJustBrownian
        bHeatEq = True


    #can be purebrownian or modpool brownian(in that case particle number will be constant, but we'll add these in anyway), or pure (stands for pure two-sided multinomial Brownian-Huygens, modpool; boson or fermi)")
    elif opts.dynamics == 'pure' or opts.dynamics == 'purebrownianhuygens':
        argdict['InOutFunction'] = PureBHP
    elif opts.dynamics == 'modpool':
        argdict['InOutFunction'] = PureModPool
    

    # the rest of the che choices for dynamics necessitate Z-tokens
    elif opts.dynamics in ('boson', 'bose', 'bose-einstein', 'boseeinstein'):
        argdict['InOutFunction'] = PedanticKeepGoing # PedanticKeepGoing #KeepGoing # PedanticPedantic if bPedantic else AlternateToPedantic # AlternateToPedantic
        CoreFn = ModPool
        argdict['CoreFn'] = PureModPool
        if opts.limit != 0:
            print("Cannot specify bose dynamics with the --limit switch.")
            sys.exit()



    elif opts.dynamics in ('fermion', 'fermi', 'fermi-dirac', 'fermidirac','pauli'):
        argdict['InOutFunction'] = PedanticKeepGoing # Pedantic if bPedantic else AlternateToPedantic   
        if opts.limit > 1:
            if not(opts.dynamics != 'gentile'):
                print("Given that the --limit switch  is", opts.limit, " G. Gentile's generalization of the Pauli exclusion principle will be assumed.") 
        
        elif opts.limit == 0:
            bSpecifiedLimit = False
            for iswitch in sys.argv:
                if iswitch in ["-m", "--limit", "--particlelimit"]:
                    bSpecifiedLimit = True
                    break
            if not(bSpecifiedLimit):
                longtxt = "\nYou have specified Fermi dynamics with no limit, so the limit is assumed to be 1 (and for legacy reasons, the spreading algorithm used in the code will slightly different than the one used when --limit has been set to 1, but the expected values will be the same in either case."
                print(textwrap.fill(longtxt, width=maxtxtlen))
                print(" ")
                opts.limit = 1
                CoreFn = FermiOut6
                argdict['CoreFn'] = FermiOut6
            else:
                CoreFn = FermiOutM
                argdict['CoreFn'] = FermiOutM
    
            argdict['CoreFn'] = FermiOut6
        else:
            argdict['CoreFn'] = BoseOutM
    else:
        print("WRONG DYNAMICS -- try again...see Iterate() for a list of all acceptable choices.")
        import pdb; pdb.set_trace()




    #correlwatch = correlwatch_t()
    #comparison = None
    myseed = abs( opts.seed )
    
    # for reasons beyond me, both these seeds have to be set or results will not replicate:
    # see https://stackoverflow.com/questions/46661426/why-random-seed-does-not-make-results-constant-in-python
    # and also https://stackoverflow.com/questns/31057g/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    # also, try using
    rn.seed(myseed)
    np.random.seed( myseed )

    ggraph = CreateGraph(opts)

    #ggraph.Create2StepPaths()

    if opts.dynamics in ('pauli', 'fermion', 'fermi', 'fermi-dirac', 'fermidirac','whorl'):
        ggraph.bNeedZTokens = True    


    CUTOFF = opts.CUTOFF
    CUTOFFVOLUME = opts.CUTOFFVOLUME
    argdict['Cutoff'] = CUTOFF
    argdict['CutoffVolume'] = CUTOFFVOLUME
    
    probtracker = growthprob_t(1, ggraph)
    FAC = opts.regprob
    if CUTOFF != 0:
        probtracker.M = int(np.ceil(CUTOFF))

    if (FAC < 1.0 or CUTOFF != 0) and not(opts.dynamics in ['pure', 'purebhp', 'modpool', 'bose']):
        print("ERROR: the --CUTOFF and --regprob options can be used only for 'pure' and 'modpool' dynamics.  ")
        print("(specifically, the cutoff parameter dictates how high an amplitude must be before regularization is engaged,  ")
        print("while if regprob is < 1.0 -- say 0.25 -- then any possible attempt at regularization will be skipped 75%  of the time")
        sys.exit()

    PARTICLE_LIMIT = opts.limit # for fermions, this shouldf be set to 1       
    argdict['Limit'] = opts.limit

    totNarc = np.sum([len(ggraph.NodeVec[i].Neighbors) for i in range(ggraph.NNode)])
    print("Graph has", ggraph.NNode, "nodes and", np.sum(totNarc), "edges")

    tarr = []
    absarr = [] # integrate the total particle count
    absarrz = [] # integrate the total number of tokens/antitokens in the space
    l2arr = [] # integrate the square of particle count at every arc
    redarr = []


    maxarr = []
    minarr = []
    

    sumx = []
    sumx2 = []


    if opts.dynamics == 'bernoulli':
        sumamparr = []
        sumamparr2 = []

    prevthisarr = None
    TMAX = opts.steps
    NRuns = opts.nruns


    bDoContinuous = True
    bGenericInitializationRoutine = False # use this for random graphs which are to be initialized elsewhere
    if opts.loadpicklefile != '':
        bDoContinuous = False
        bGenericInitializationRoutine = True # use this for random graphs which are to be initialized elsewhere



    NRuns_needed_to_fit_control = 0
    NRuns_needed_to_fit_test = 0
    
    if bDoContinuous:
        if not(bGenericInitializationRoutine):
            
            if BoundaryConditions == "":
                ggraph = Initialize(ggraph, opts)
                thisarrcont = ExportDimGraph(ggraph, True)
            else:
                centernode = None
                if ggraph.NDim > 0:
                    centernode = tuple([opts.length//2-1] * opts.dimension)
                InitializerFn(ggraph, opts, centernode, True)

        else:
            ggraph = Initialize(ggraph, opts, True)

       # if ggraph.NDim > 0 and (ggraph.TorLen % 2 == 0):
        #    if not(ggraph.bBipartiteParityCheck()):
        #        print("WARNING: For bipartite graphs (e.g. rectuangular arrays), it is strongly recommended to ")


        if TMAX < 200: # totally arbitrary; if TMAX >= 200, we assume that we're not going to be doing a convergence study and therefore can dispense with the continuous case
            for t in range(TMAX):
                Iterate(ggraph, opts, argdict, True)
            if ggraph.NDim > 0:
                thisarrcont = ExportDimGraph(ggraph, True)
        else:
            print("Skipping the calculation of the continuous-case evolution.")
            #if ggraph.NDim > 0:
            #    thisarrcont =  ExportDimGraph(ggraph, True)
            

        if ggraph.NDim > 0:
            sumamparraycont = np.sum(thisarrcont, ggraph.NDim)
            sumamparray = np.zeros( tuple([opts.length] * opts.dimension) )
            sumamparrayctl = copy(sumamparray)
            sumamparrayx = copy(sumamparray)
            sumamparrayz = copy(sumamparray)
            sumamparraye = copy(sumamparray)
            #sumamparrayzin = copy(sumamparray) 
            sumampall = np.zeros( tuple([opts.length] * opts.dimension + [2*opts.dimension]) )

            sumampallz = copy(sumampall)
            #sumampallzin = copy(sumampall)
            sumampallx = copy(sumampall)
            sumampalle = copy(sumampall)
            sumampallctl = copy(sumampall)  
        else:
            sumamparraycont = [np.sum(inode.ContAmplitude) for inode in ggraph.NodeVec]
            sumamparray = np.zeros( ggraph.NNode ).astype("int")
            sumamparrayctl = copy(sumamparray)
            sumamparrayx = copy(sumamparray)
            sumamparrayz = copy(sumamparray)
            sumamparraye = copy(sumamparray)
            #sumamparrayzin = copy(sumamparray) 
            sumampall = CreateZerosFromGraph(ggraph)

  


    bFullPrint = opts.print and opts.dimension == 2

    bAltErr = True
    if bAltErr:
        alterr = []
        alterr2 = [] 
    
    """
    f = open('blah.pkl', 'rb')
    resout = pickle.load(f)
    irun0 = resout[0]
    sumamparray = resout[1]
    sumampallctl = resout[2]
    # f = open('blah.pkl','wb'); pickle.dump((irun, sumamparray, sumampallctl, sumampall, sumampallctl), f)
    """

    

    for irun in range(NRuns):
        if not(bGenericInitializationRoutine):
            if BoundaryConditions == "":
                ggraph = Initialize(ggraph, opts)
                thisarrcont = ExportDimGraph(ggraph, True)
            else:
                centernode = None
                if ggraph.NDim > 0:
                    centernode = tuple([opts.length//2-1] * opts.dimension)
                InitializerFn(ggraph, opts, centernode)
        else:
            ggraph = Initialize(ggraph, opts)

        
        """
        if not(bGenericInitializationRoutine):
            if BoundaryConditions == "":
                centernode = tuple([opts.length//2-1] * opts.dimension)
            else:
                InitializerFn(ggraph, opts, centernode)
        else:
            if opts.loadpicklefile == '':
                Initialize(ggraph, opts)
        """



        thisabs = 0
        thisl2 = 0
        for i in range(ggraph.NNode):
            thisabs += np.sum(np.abs(ggraph.NodeVec[i].Amplitude))
            thisl2 += np.sum([(x * x) for x in ggraph.NodeVec[i].Amplitude])

        if irun == 0:
            print("Initialized with sumabs ", str(thisabs), " and suml2 ", thisl2)


    
        if ggraph.NDim > 0: 
            centernode = tuple([opts.length//2-1] * opts.dimension)
            #lo = 0 # centernode[0] - 1
            #hi = 4 # centernode[0] + 3
            #loy = 0 # enternode[0] - 3
            #hiy = 4 # centernode[0] + 1

            
            grid = ExportDimGraph(ggraph)#[lo:hi,loy:hiy,:] # [12:20,12:20,:]
            if ggraph.bNeedZTokens:
                gridz = ZExportDimGraph(ggraph)#[lo:hi,loy:hiy,:]

            gridx = CtlExportDimGraph(ggraph)#[lo:hi,loy:hiy,:]
                
            thisarr = ExportDimGraph(ggraph)
            if bFullPrint:
                ggraph.t = 0
                print("BEGIN  t=0")
                if ggraph.TorLen > 4:
                    print("WARNING: the --print option is best suited for a --lenght parameter of 4, and only only a 4x4 section of this grid will be printed. Inspect code for details.")

                print("++++reg")
                print4grid(grid, ggraph.t)
                
                if ggraph.bNeedZTokens:
                    print("+++++++++++++++++++++Z")
                    print4grid(gridz, ggraph.t)



                print("+++++++++++++++++++++Ctl")
                print4grid(gridx, ggraph.t)
                print("+++++++++++++++++++++")
                print("+++++++++++++++++++++")
        else:
            thisarr = CreateZerosFromGraph(ggraph)




                 

                    





        bGetPctBreach = True
        excpct = []

        
        for t in range(TMAX):          
            if (t % 100 == 0) and len(tarr) >= 100:
                if ggraph.bNeedZTokens:
                    start = NonZeroSection(absarrz)
                    
                    if len(tarr) - start > 20:
                        print("Run,Time", irun, t, linregress(np.log(1+np.array(tarr[start:])), np.log(absarrz[start:])))
                    else:
                        print("Run,Time", irun, t, "...")
                else:
                    start = NonZeroSection(absarr)
                    if len(tarr) - start > 20:
                        print("Run,Time", irun, t, linregress(np.log(1+np.array(tarr))[start:], np.log(absarr)[start:]))
                    else:
                        print("Run,Time", irun, t, "...")
            ggraph.t = t
            ggraph.irun = irun
            # you can shuffle the order update  with each step, but I won't bother
    
            #if (irun,t) == (2,2):
            #    print("here")
            #    import pdb; pdb.set_trace()
            
            Iterate(ggraph, opts, argdict)
            
            tarr_print_increment = 1 # 5000# 10
            tarr_pause_increment = 10_000_000_000_000_000 # never
            if not(bFullPrint):
                if t % tarr_print_increment == 0 and NRuns == 1:
                    thisabs = 0
                    thisabsz = 0
                    thisl2 = 0
                    for i in range(ggraph.NNode):
                        thisabs += np.sum(np.abs(ggraph.NodeVec[i].Amplitude))
                        thisl2 += np.sum([x * x for x in ggraph.NodeVec[i].Amplitude])
                        thissuml2 = np.sum(thisl2)


                    if ggraph.bNeedZTokens:
                        thissumx, thissumx2 = ggraph.SumxZ(False)
                    else:
                        thissumx, thissumx2 = ggraph.Sumx(False)
                    sumx.append(thissumx)
                    sumx2.append(thissumx2)
                    tarr.append(t)
                    redarr.append(argdict["Reduction"]/float(tarr_print_increment))
                    argdict["Reduction"] = 0
                    l2arr.append(thisl2)
                    absarr.append(thisabs)
                    absarrz.append(ggraph.SumAbsZ(False))

                    probtracker.UpdateXarr()   

                    if bGetPctBreach:
                        Mb = 4
                        Narc = np.sum([len(inode.Amplitude) for inode in ggraph.NodeVec])
                        Nexcess = np.sum([np.array(np.abs(np.array(inode.Amplitude) > Mb)).astype("int") for inode in ggraph.NodeVec])
                        excpct.append(Nexcess/float(Narc))
                        #excpct.append( np.std([np.array(inode.Amplitude) for inode in ggraph.NodeVec]) )

                    minarr.append(np.min(thisarr))
                    maxarr.append(np.max(thisarr))

                    if opts.dynamics == 'bernoulli':
                        sumamp = SumAmp(ggraph)
                        sumamparr.append(sumamp)
                        sumamparr2.append(np.abs(sumamp))
                    
                    
                    #if t % tarr_pause_increment == 0:
                    #    SaveGraph(ggraph, "longfile.pkl")
                    #    print("absarr last batch", t, absarr[-(tarr_pause_increment//tarr_print_increment):])

            if bFullPrint:# and t > 0:

                centernode = tuple([opts.length//2-1] * opts.dimension)
                centernode = tuple([opts.length//2-1] * opts.dimension)
                lo = 0 # centernode[0] - 1
                hi = 4 # centernode[0] + 3
                loy = 0 # centernode[0] - 3
                hiy = 4 # centernode[0] + 1
                grid = ExportDimGraph(ggraph)[lo:hi,loy:hiy,:] # [12:20,12:20,:]
                gridz = ZExportDimGraph(ggraph)[lo:hi,loy:hiy,:]
                gridx = CtlExportDimGraph(ggraph)[lo:hi,loy:hiy,:]

                # ExportDimGraphNodeNumber(ggraph)[14:18,14:18]

                if False:

                    print4grid(grid, ggraph.t+1)
                    if ggraph.bNeedZTokens:
                        print("+++++++++++++++++++++")
                        print4grid(gridz, ggraph.t+1)
                        print("+++++++++++++++++++++")
                        print4grid(gridx, ggraph.t+1)
                    print("+++++++++++++++++++++")
                    print("+++++++++++++++++++++")
                
                if True:
                    print("+++++++++++++++++++++Reg  t=", t, "this is post-iteration, so t is effectively ", t+1)
                    print4grid(grid, ggraph.t+1)
                    if ggraph.bNeedZTokens:
                        print("+++++++++++++++++++++Z")
                        print4grid(gridz, ggraph.t+1)

                    print("+++++++++++++++++++++Ctl")
                    print4grid(gridx, ggraph.t+1)
                    print("+++++++++++++++++++++")
                    print("+++++++++++++++++++++")

                    import pdb; pdb.set_trace()
                
            #import pdb; pdb.set_trace()



            if (t != 0) and (t % 250000 == 0):
                print("Run,Time", irun, t)
            




        #import pdb; pdb.set_trace()
        #print(t, "sumamp", ggraph.SumAmp(0), "sumampZ", ggraph.SumAmpZ(0), "sumampE", ggraph.SumAmpE(0), ggraph.SumAbs(0), ggraph.SumAbsZ(0), ggraph.SumAbsE(0))

        if ggraph.NDim > 0:

            thisarr = ExportDimGraph(ggraph)
            thisarrctl = CtlExportDimGraph(ggraph)
            if ggraph.bNeedZTokens:
                thisarrz = ZExportDimGraph(ggraph)
                thisarrx = ExportDimGraphx(ggraph)
                thisarrzin = ZInExportDimGraph(ggraph)

            else:
                thisarrx = thisarr

            # the rest doesn't really get used, but probably will at some point, so it stays.
            sumamparray += np.sum(thisarr, ggraph.NDim)
            sumamparrayctl += np.sum(thisarrctl, ggraph.NDim)
            if ggraph.bNeedZTokens:
                sumamparrayz += np.sum(thisarrz, ggraph.NDim)
                #sumamparrayzin += np.sum(thisarrzin, ggraph.NDim)
                sumamparrayx  += np.sum(thisarrx, ggraph.NDim)
                
            else:
                sumamparrayx = sumamparray

            sumampall += thisarr
            if ggraph.bNeedZTokens:
                sumampallx += thisarrx
                sumampallz += thisarrz
                #sumampallzin += thisarrzin
            else:
                sumampallx = sumampall

            sumampallctl += thisarrctl
        else:
            thisarr = [np.sum(inode.Amplitude) for inode in ggraph.NodeVec]
            thisarrctl = [np.sum(inode.AmplitudeCtl) for inode in ggraph.NodeVec]
            if ggraph.bNeedZTokens:
                thisarrz = [np.sum(inode.ZAmplitude) for inode in ggraph.NodeVec]
                #thisarrzin = [np.sum(inode.AmplitudeZIn) for inode in ggraph.NodeVec]
                thisarrx = [(thisarr[j] - thisarrz[j]) for j in range(ggraph.NNode)]

            
            sumamparray += thisarr
            sumamparrayctl += thisarrctl
            if ggraph.bNeedZTokens:
                sumamparrayz += thisarrz
                #sumamparrayzin += thisarrzin
                sumamparrayx += thisarrx

            # the rest doesn't really get used, but probably will at some point, so it stays.
            #sumampall = AddByGraphNeighbors(sumampall, thisarr)
            #sumampallctl = AddByGraphNeighbors(sumampallctl, thisarrctl)
            #sumampallz = AddByGraphNeighbors(sumampallz, thisarrz)
            ##sumampallzin = AddByGraphNeighbors(sumampallzin, thisarrzin)
            #sumampallx = AddByGraphNeighbors(sumampallx, thisarrx)

            
        if NRuns > 1 and irun > 0 and (TMAX < 100 and irun % 10 == 0 or irun == NRuns-1):
            #sumamparraycont[:,:]
            #np.round(np.sum((sumampallctl/float(irun+1))[:,:,:], ggraph.NDim),3)
            if ggraph.NDim > 0:
                errmat = (sumamparrayx/float(irun+1)) - sumamparraycont
                errmat2 = errmat * errmat
                errmatctl = np.sum((sumampallctl/float(irun+1)), ggraph.NDim) - sumamparraycont
                errmatctl2 = errmatctl * errmatctl


                if bAltErr:
                    errmatalt = sumamparray - (irun+1)*sumamparraycont
                    summalterr = np.sum(errmatalt)
                    summalterr2 = np.sum(errmatalt*errmatalt)
                    alterr.append(summalterr)
                    alterr2.append(summalterr2)   


                chisq = np.sum(errmat2)
                chisqctl = np.sum(errmatctl2)

            else:
                thiserr = np.array(sumamparrayx)/float(irun+1) - np.array(sumamparraycont)
                thiserrctl = np.array(sumamparrayctl)/float(irun+1) - np.array(sumamparraycont)
                chisq = np.sum(thiserr * thiserr)
                chisqctl = np.sum(thiserrctl * thiserrctl)


            if bHeatEq:
                print("Run", irun, ' '.join(sys.argv), " :: MonteCarlo_sumsqerr", chisq)
            else:
                print("Run", irun, ' '.join(sys.argv), " :: BHP_err", chisq, " MP_err", chisqctl )

            import pdb; pdb.set_trace()

            """
            bStopWhenChiSqIsSmall = False
            if bStopWhenChiSqIsSmall:
                if  NRuns_needed_to_fit_control == 0:
                    if chisqctl < 0.001:
                        NRuns_needed_to_fit_control = copy(irun)
                
                if  NRuns_needed_to_fit_test == 0:
                    if chisq < 0.001:
                        NRuns_needed_to_fit_test = copy(irun)
                
                if NRuns_needed_to_fit_control > 0 and NRuns_needed_to_fit_test > 0:
                    print("NRuns_needed_to_fit_control", NRuns_needed_to_fit_control, "NRuns_needed_to_fit_test", NRuns_needed_to_fit_test, ' '.join(sys.argv))
                    break
            """


            

            if (irun % 500 == 0 or irun == NRuns-1) and ggraph.NDim == 2 and ggraph.TorLen <= 8 :
                print("floatingpt (true) solution")
                print(np.round(sumamparraycont,3).T)
                print("\n")
                if not(bHeatEq):
                    print("modpool MonteCarlo result")
                    print(np.round(np.sum((sumampallctl/float(irun+1))[:,:,:],2),3).T)
                    print("\n")
                if ggraph.bNeedZTokens:
                    print("MonteCarlo result of user-specified dynamics (using Z-tokens and whorl-regularization)")
                else:
                    print("MonteCarlo result of user-specified dynamics ")
                print(np.round(np.sum((sumampallx/float(irun+1))[:,:,:],2),3).T)
                #print("bounded NabsX (poorer chisq)")
                #print(np.round(np.sum((sumampallx/float(irun+1))[:,:,:],2),4).T)
                print("\n")

    

        
    #correlwatch.corrcoeff()
        
    #(sumamparray/float(NRuns))[12:20,12:20]
    # sumamparraycont[12:20,12:20]
    #errmat = (sumamparray/float(NRuns)) - sumamparraycont
    #errmat2 = errmat * errmat
    #chisq = np.sum(errmat2)

    #import pdb; pdb.set_trace()

    bGraphics = TMAX >= 1000

    if bGraphics:
        start = NonZeroSection(absarr)
        
        if ggraph.bNeedZTokens:

            if len(absarr) - start > 20:
            
                plt.plot(tarr[start:], absarr[start:])
                titl = opts.dynamics.capitalize() + (' total particle count')
                plt.title(titl)
                plt.show()
            a = np.log(1 + np.array(absarrz))
            start = NonZeroSection(absarrz)
   
            if len(tarr) - start > 20:
                lr = linregress(np.log(tarr[start:]), np.log(absarrz[start:])  )
                plt.plot(tarr[start:],absarrz[start:])      
                titl = opts.dynamics.capitalize()  + (' total Z-token count (y~t**%.2f)' % (lr.slope,))
            else:
                print("Skipping plots -- not enough non-zero values")

            
            #plt.plot(tarr, absarr, color='blue', label="Total (regular) particle count")
            #plt.plot(tarr,absarrz, color='red', label="Total Z-token count")
            #plt.xlim(0, 250)
            #plt.ylim(0,1500)
            #plt.legend()
            #import pdb; pdb.set_trace()
            #plt.show()




        else:
            start = NonZeroSection(absarr)
   
            if len(tarr) - start > 20:
                plt.plot(tarr[start:],absarr[start:])
                lr = linregress(np.log(tarr[start:]), np.log(absarr[start:])  )
                titl = opts.dynamics.capitalize()  + (' total particle count (y~t**%.2f)' % (lr.slope,))
                plt.title(titl)
                plt.show()
                a = np.log(1 + np.array(absarrz))
                start = NonZeroSection(absarrz)

                if len(tarr) - start > 20:
                    plt.plot(tarr[start:],absarrz[start:])      
                    titl = opts.dynamics.capitalize()  + (' total Z-token count (y~t**%.2f)' % (lr.slope,))            
            
    
        
        farr = np.exp(lr.intercept + lr.slope * np.log( tarr[start:] ))
        plt.plot(tarr[start:],farr)
        
        plt.title(titl)
        plt.show()

        #import pdb; pdb.set_trace()
        #print("For args", ' '.join(sys.argv))
        #print(linregress(a[start:], b[start:]))
        #interceptwhenslopeis1 = np.mean(b) - np.mean(a)
        #print("First do Integ(x*x), for args", ' '.join(sys.argv))
        #print("intercept if slope is 1.0", interceptwhenslopeis1, np.exp(interceptwhenslopeis1), "sqrt", np.sqrt(np.exp(interceptwhenslopeis1)))
        #interceptwhenslopeis1half = np.mean(b) - 0.5*np.mean(a)
        #print("intercept if slope is 0.5", interceptwhenslopeis1half, np.exp(interceptwhenslopeis1half), "sqrt", np.sqrt(np.exp(interceptwhenslopeis1half)))




        #import pdb; pdb.set_trace()

        #a = np.log(1 + np.array(tarr))
        #if opts.dynamics == 'oneplus':
        #    a = 1 + np.array(tarr)
        #if ggraph.bNeedZTokens:
        #    c = np.log(1.0 + np.array(absarrz))
        #else:
        #    c = np.log(1.0 + np.array(absarr))
        #titl = opts.dynamics + ' integ(abs(x))'

        #plt.plot(a,c)
        #plt.title(titl)
        #plt.show()


        #from scipy.stats import linregress
        #print(linregress(a, c))
        
        #interceptwhenslopeis1 = np.mean(c) - np.mean(a)
        #print("Then do integ(x_coord*x_coord), for args", ' '.join(sys.argv))
        #a = np.log(1 + np.array(tarr))
        #if opts.dynamics == 'oneplus':
        #    a = 1 + np.array(tarr)
        # = np.log(1.0 + np.array(sumx2))
        #titl = opts.dynamics + ' integ(x_coord*x_coord)'

        #plt.plot(a,g)
        #plt.title(titl)
        #plt.show()


        #from scipy.stats import linregress
        #print(linregress(a, g))
        
        #interceptwhenslopeis1 = np.mean(c) - np.mean(a)
        #print("Then do integ(abs(x)), for args", ' '.join(sys.argv))
        #print("intercept if slope is 1.0", interceptwhenslopeis1, np.exp(interceptwhenslopeis1), "sqrt", np.sqrt(np.exp(interceptwhenslopeis1)))
        #interceptwhenslopeis1half = np.mean(c) - 0.5*np.mean(a)
        #print("intercept if slope is 0.5", interceptwhenslopeis1half, np.exp(interceptwhenslopeis1half), "sqrt", np.sqrt(np.exp(interceptwhenslopeis1half)))




if __name__ == '__main__':   
    Main()




"""

f = open('blahbose_20260211.pkl', 'wb'); pickle.dump((tarr, absarr, l2arr), f); f.close()

os.chdir("/Users/hrvojehrgovcic/quant/lattice_datasets_for_papers/ZTokenEvolution/")

import pickle
f = open('blah.pkl', 'rb')
ggrid = pickle.load(f)
f.close()

f = open('blah3.pkl', 'rb')
coords,nodes,coordpath = pickle.load(f)
f.close()


#####
f = open('blahpure.pkl', 'rb'); 
blobpure = pickle.load(f)
tarr = blobpure[0]; absarrpure = blobpure[1]
f.close()

f = open('blahmodp.pkl', 'rb'); 
blobmodp = pickle.load(f)
tarr = blobmodp[0]; absarrmodp = blobmodp[1]
f.close()

f = open('blahbose_20260209.pkl', 'rb'); 
blobbose = pickle.load(f)
tarr = blobbose[0]; absarrbose = blobbose[1]
f.close()

#plt.plot(tarr,absarrpure)
#plt.plot(tarr,absarrmodp)
#plt.plot(tarr,absarrbose)

lim = 10000
plt.plot(tarr[:lim],absarrpure[:lim], color='r', label="Pure BHP (linear growth)")
plt.plot(tarr[:lim],absarrmodp[:lim], color='blue', label="ModPool BHP (sqrt(T) growth)")
#plt.plot(tarr[:lim],absarrbose[:lim], color='green', label="Whorl-regularization with M=0 (stable)")
plt.ylim(0, 10000)
plt.xlim(0,lim + 50)

plt.xlabel('T')
plt.ylabel('N_particles + N_antiparticles')
plt.title("Comparing particle growth with different dynamics")
plt.legend()
plt.show()


#####




"""