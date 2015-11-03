#!/usr/bin/env python
"""Simple example for showing surface mesh functionality
   Variables that can be set in mat3d.mesh :
   colors = Colors   - Sets the colormap
   linewidth = 0.1   - Sets the width of the plotlines
   precision=3       - Sets the number of digits accuracy for the text on the axis    
   num_ticks=5       - Number of ticks on axes, evenly spaced 
   fill_mesh = 0     - If one, fills the mesh with the colormap in colors"""

import numpy as N
import mat3d as M
from matplotlib import numerix as nu

u = N.linspace(0, 2*N.pi,20)
v = N.linspace(0, N.pi,20)
        
x=nu.outerproduct(N.cos(u),N.sin(v))
y=nu.outerproduct(N.sin(u),N.sin(v))
z=nu.outerproduct(N.ones(N.size(u)), N.cos(v))

# call the actual plotting routine
x = M.mesh(x,y,z,fill_mesh=1,precision=1,num_ticks = 3)




