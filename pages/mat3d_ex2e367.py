#!/usr/bin/env python
"""Simple example for showing mat3d.plot3 functionality
   Variables that can be set in mat3d.plot3 :
   colors = Colors   - Sets the colormap
   linewidth = 0.1   - Sets the width of the plotlines
   precision=3       - Sets the number of digits accuracy for the text on the axis    
   num_ticks=5       - Number of ticks on axes, evenly spaced """

import numpy as N
import mat3d as M

x0,x1,npts_x = -1.0,1.1,11 
y0,y1,npts_y = -1.0,1.1,11 

X,Y = N.mgrid[x0:x1:0.1,y0:y1:0.1]
Z = X * N.exp(-X**2 - Y**2)

M.plot3(X,Y,Z,linewidth =2.0)
