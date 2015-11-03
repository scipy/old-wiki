#!/usr/bin/env python
"""Simple example for showing surface mesh functionality, this example uses default colormap in reverse
   Variables that can be set in mat3d.mesh :
   colors = Colors   - Sets the colormap
   linewidth = 0.1   - Sets the width of the plotlines
   precision=3       - Sets the number of digits accuracy for the text on the axis    
   num_ticks=5       - Number of ticks on axes, evenly spaced 
   fill_mesh = 0     - If one, fills the mesh with the colormap in colors"""

import numpy as N
import mat3d as M

# Set mesh configuration
x0,x1,npts_x = -1.0,1.1,11 
y0,y1,npts_y = -1.0,1.1,11 

# N.mgrid requires the stepsize in the call
x_delta = (x1-x0)/float(npts_x)
y_delta = (y1-y0)/float(npts_y)

X,Y = N.mgrid[x0:x1:0.1,y0:y1:0.1]
Z = X * N.exp(-X**2 - Y**2)

NewCl = N.array([[172.0, 201.0,86.0],[115.0, 201.0,86.0],[86.0,201.0,115.0],[86.0, 201.0,172.0],[86.0, 172.0,201.0],[86.0,115.0,201.0],[115.0, 86.0,201.0],[172.0, 86.0,201.0],[201.0, 86.0,172.0],[201.0,86.0,89.0],[201.0, 115.0,86.0],[201.0, 172.0,86.0]])/256
x = M.mesh(X,Y,Z,NewCl, fill_mesh=1)
