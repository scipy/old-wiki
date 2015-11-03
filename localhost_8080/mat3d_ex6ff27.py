#!/usr/bin/env python
"""Simple example for showing mat3d.plot3_points functionality
   Variables that can be set in mat3d.plot3_points :
   colors = Colors   - Sets the colormap
   linewidth = 0.1   - Sets the width of the plotlines
   precision=3       - Sets the number of digits accuracy for the text on the axis    
   num_ticks=5       - Number of ticks on axes, evenly spaced 
   pointsize=5       - Sets the size of points"""

import numpy as N
import mat3d as M

A = N.array(N.random.rand(500,3)) 
M.plot3_points(A,precision=1)
