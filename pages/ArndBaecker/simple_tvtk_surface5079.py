from enthought.tvtk.tools import mlab
from Numeric import arange, cos, sqrt

gui = mlab.GUI()
fig = mlab.figure()
x = arange(-5.0, 5.0, 0.2)
y = arange(-5.0, 5.0, 0.2)

def fct(x, y):
    return cos( sqrt(x**2+y**2))

s = mlab.SurfRegular(x, y , fct)
fig.add(s)
gui.start_event_loop()
