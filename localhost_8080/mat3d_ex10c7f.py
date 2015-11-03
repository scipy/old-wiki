#!/usr/bin/env python
"""Simple example for showing mat3d.plot3 functionality
   Variables that can be set in mat3d.plot3 :
   colors = Colors   - Sets the colormap
   linewidth = 0.1   - Sets the width of the plotlines
   precision=3       - Sets the number of digits accuracy for the text on the axis    
   num_ticks=5       - Number of ticks on axes, evenly spaced """

import numpy as N
import mat3d as M

t = N.linspace(0,10*N.pi,500)

# call the actual plotting routine
M.plot3(N.sin(t),N.cos(t),t,linewidth=2)
