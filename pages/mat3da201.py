"""OpenGL-based 3d surface plot"""
#last updated 12/10/2006
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.Tk import *

import numpy as N
import tkFileDialog
# global constants
Colors = N.array([[201.0, 172.0,86.0],[201.0, 115.0,86.0],[201.0,86.0,89.0],
                [201.0, 86.0,172.0],[172.0, 86.0,201.0],[115.0, 86.0,201.0],
                  [86.0,115.0,201.0],[86.0, 172.0,201.0],[86.0, 201.0,172.0],
                  [86.0,201.0,115.0],[115.0, 201.0,86.0],[172.0, 201.0,86.0]])/256

glutInit(sys.argv)
                  
class figure :
    
    def __init__(self):
        return

    def PutText3d(self,x,y,z,text) :
        """ Draws text (using GLUT package) at given x,y,z coordinates"""
        glRasterPos3f(x,y,z)
        for i in range(len(text)) :
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(text[i]));
  
    def SetColor(self,height) :
        """ Sets the colour from the colour index according to the plot height"""
        color_index = int(round((height)*11))
        glColor3f(self.Colors[color_index,0],self.Colors[color_index,1],
        self.Colors[color_index,2])

    def Rescale(self,X,Y,Z) :
        """Scales values from 0 to 1"""
        
        X,Y,Z = map(N.asarray,(X,Y,Z))
        
	self.xmin = X.min()
        self.xmax = X.max()
        self.ymin = Y.min()
        self.ymax = Y.max()
        self.zmin = Z.min()
        self.zmax = Z.max()

	#rescaling
        if self.xmax==self.xmin :
            xdif = 0.01
        else:
            xdif = abs(self.xmax-self.xmin)

        if self.ymax==self.ymin :
            ydif = 0.01
        else:
            ydif = abs(self.ymax-self.ymin)

        if self.zmax==self.zmin :
            zdif = 0.01
        else:
            zdif = abs(self.zmax-self.zmin)
	#rescaling
        self.X = (X-self.xmin)/xdif
        self.Y = (Y-self.ymin)/ydif
        self.Z = (Z-self.zmin)/zdif


    def InitGfx(self) :
        glShadeModel(GL_SMOOTH)
        glClearColor(1, 1, 1, 1)
        glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
        for option in [GL_DEPTH_TEST,GL_LINE_SMOOTH,GL_POINT_SMOOTH, \
                       GL_POLYGON_SMOOTH,GL_BLEND]:
            glEnable(option)
            
        glDepthFunc(GL_LEQUAL);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        
        for hint in [GL_PERSPECTIVE_CORRECTION_HINT,GL_LINE_SMOOTH_HINT, \
                     GL_POLYGON_SMOOTH_HINT]:
            glHint(hint,GL_NICEST)
            
        glClearColor(1, 1, 1, 0)    
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)    

    def DrawPlotBox(self) :
        """ Draws 3 faces of a box that will contain our plot"""
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        modelview = N.array(modelview)
	t_model = modelview.transpose()
        inv_modelview = N.linalg.inv(t_model)
        self.inv_mv = inv_modelview
        eye = N.array([0,0,-1,1])
        view = N.dot(inv_modelview,eye)
        view = view/float(view[3])
        view = view[0:3]  
        self.view = view
        init_x = N.array([1,0,0,1])
        new_x = N.dot(inv_modelview,init_x)
        new_x = new_x/float(new_x[3])
        new_x= new_x[0:3] 
        self.new_x = new_x
        init_y = N.array([0,1,0,1])
        new_y = N.dot(inv_modelview,init_y)
        new_y = new_y/float(new_y[3])
        new_y= new_y[0:3]
        self.new_y = new_y

        offset = self.offset
        
        glColor3f(0.98, 0.98, 0.98)
        glDisable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
        glBegin(GL_QUADS) #x,z,y
        
        if N.dot(view,[1,0,0]) < 0 :
            glVertex3f(0-0.5-offset ,0-0.5-offset ,0-0.5-offset )       #1
            glVertex3f(0-0.5-offset ,1-0.5+offset,0-0.5-offset )
            glVertex3f(0-0.5-offset ,1-0.5+offset,1-0.5+offset  )
            glVertex3f(0-0.5-offset ,0-0.5-offset ,1-0.5+offset  )
            self.sides = [1]
        else :  
            glVertex3f(1-0.5+offset  ,0-0.5-offset ,0-0.5-offset )               #2
            glVertex3f(1-0.5+offset   ,0-0.5-offset ,1-0.5+offset )
            glVertex3f(1-0.5+offset,1-0.5+offset ,1-0.5+offset )
            glVertex3f(1-0.5+offset  ,1-0.5+offset  ,0-0.5-offset )
            self.sides = [2]

        if N.dot(view,[0,1,0]) < 0 :

            glVertex3f(0-0.5-offset,0-0.5-offset,0-0.5-offset)       #3
            glVertex3f(0-0.5-offset,0-0.5-offset,1-0.5+offset)
            glVertex3f(1-0.5+offset ,0-0.5-offset,1-0.5+offset)
            glVertex3f(1-0.5+offset,0-0.5-offset ,0-0.5-offset )
            self.sides.append(3)
        else :
            glVertex3f(0-0.5-offset ,1-0.5+offset ,0-0.5-offset )    #4
            glVertex3f(1-0.5+offset ,1-0.5+offset ,0-0.5-offset )
            glVertex3f(1-0.5+offset,1-0.5+offset ,1-0.5+offset )
            glVertex3f(0-0.5-offset ,1-0.5+offset,1-0.5+offset)
            self.sides.append(4)

        if N.dot(view,[0,0,1]) > 0 :
            glVertex3f(0-0.5-offset ,0-0.5-offset ,1-0.5+offset)               #6
            glVertex3f(0-0.5-offset ,1-0.5+offset,1-0.5+offset )
            glVertex3f(1-0.5+offset,1-0.5+offset,1-0.5+offset)
            glVertex3f(1-0.5+offset,0-0.5-offset ,1-0.5+offset)
            self.sides.append(6)
        else : 
            glVertex3f(0-0.5-offset,0-0.5-offset ,0-0.5-offset )    #5
            glVertex3f(1-0.5+offset,0-0.5-offset ,0-0.5-offset )
            glVertex3f(1-0.5+offset,1-0.5+offset,0-0.5-offset )
            glVertex3f(0-0.5-offset ,1-0.5+offset,0-0.5-offset )
            self.sides.append(5)         
        glEnd()
        glEnable(GL_LIGHTING)

    def DrawPlotBoxLines(self):
        """ Draws lines on the plot box"""
        glDisable(GL_LIGHTING)
        #glEnable(GL_LINE_STIPPLE)
        #glLineStipple(2,43690)
        glColor3f(0.9,0.9,0.9)
        offset = self.offset
        glBegin(GL_LINES)
        for a in range(1,4):

            if self.sides[0] == 2:
                glVertex3f(1-0.5+offset,0-0.5+0.25*a,1-0.5+offset   )
                glVertex3f(1-0.5+offset,0-0.5+0.25*a,0-0.5-offset )               #2
        
                glVertex3f(1-0.5+offset,1-0.5+offset,0-0.5+0.25*a)
                glVertex3f(1-0.5+offset,0-0.5-offset,0-0.5+0.25*a)               #2
            else :
                glVertex3f(0-0.5-offset,0-0.5+0.25*a,1-0.5+offset)
                glVertex3f(0-0.5-offset,0-0.5+0.25*a,0-0.5-offset)               #1
        
                glVertex3f(0-0.5-offset,1-0.5+offset,0-0.5+0.25*a)
                glVertex3f(0-0.5-offset,0-0.5-offset,0-0.5+0.25*a)               #1
                
            if self.sides[1] == 3:
                glVertex3f(0-0.5+0.25*a,0-0.5-offset,1-0.5+offset)
                glVertex3f(0-0.5+0.25*a,0-0.5-offset,0-0.5-offset)       #3
            
                glVertex3f(1-0.5+offset,0-0.5-offset,0-0.5+0.25*a)
                glVertex3f(0-0.5-offset,0-0.5-offset,0-0.5+0.25*a)       #3
            else :
                glVertex3f(0-0.5+0.25*a,1-0.5+offset,1-0.5+offset)
                glVertex3f(0-0.5+0.25*a,1-0.5+offset,0-0.5-offset)       #4
            
                glVertex3f(1-0.5+offset,1-0.5+offset,0-0.5+0.25*a)
                glVertex3f(0-0.5-offset,1-0.5+offset,0-0.5+0.25*a) 
                
            if self.sides[2] == 5:
                glVertex3f(1-0.5+offset,0-0.5+0.25*a,0-0.5-offset)
                glVertex3f(0-0.5-offset,0-0.5+0.25*a,0-0.5-offset)    #5
            
                glVertex3f(0-0.5+0.25*a,1-0.5+offset,0-0.5-offset)
                glVertex3f(0-0.5+0.25*a,0-0.5-offset,0-0.5-offset) #5
            else :
                glVertex3f(1-0.5+offset,0-0.5+0.25*a,1-0.5+offset)
                glVertex3f(0-0.5-offset,0-0.5+0.25*a,1-0.5+offset)    #6
            
                glVertex3f(0-0.5+0.25*a,1-0.5+offset,1-0.5+offset)
                glVertex3f(0-0.5+0.25*a,0-0.5-offset,1-0.5+offset) #6      
        glEnd()
        #glDisable(GL_LINE_STIPPLE)
        glColor3f(0,0,0)
        glEnable(GL_LIGHTING)

    def GetFormatMaxTextLength(self) :
        import decimal as D 
        D.getcontext().prec = self.precision
        ticks = []
        xticks = []
        yticks = []
        zticks = []

        for a in range(self.num_ticks):
            numberx = self.xmin +a*0.25*(self.xmax-self.xmin)
            numbery = self.ymin +a*0.25*(self.ymax-self.ymin)
            numberz = self.zmin +a*0.25*(self.zmax-self.zmin)
            strnumberx = str(D.Decimal(str(numberx))+0) 
            strnumbery = str(D.Decimal(str(numbery))+0)
            strnumberz = str(D.Decimal(str(numberz))+0)
            if ((strnumberx[0:3]=='0.0') or (strnumberx[0:4]=='-0.0')) and float(strnumberx)!=0:
                strnumberx = '%.*e' % (self.precision-1,float(strnumberx))
            if ((strnumbery[0:3]=='0.0') or (strnumbery[0:4]=='-0.0'))and float(strnumbery)!=0:
                strnumbery = '%.*e' % (self.precision-1,float(strnumbery))
            if ((strnumberz[0:3]=='0.0') or (strnumberz[0:4]=='-0.0'))and float(strnumberz)!=0:
                strnumberz = '%.*e' % (self.precision-1,float(strnumberz))
            xticks.append(strnumberx)
            yticks.append(strnumbery)
            zticks.append(strnumberz)
            ticks.append(len(strnumberx))
            ticks.append(len(strnumbery))
            ticks.append(len(strnumberz))

        return max(ticks), xticks,yticks,zticks
        

    def InsertAxesText(self) :
        """ Identifies on which sides axis should be drawn, establishes the orientation
            of the text in relation to the plot area and draws the text accordingly. """
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_MODELVIEW);
        glLineWidth(2.0)
        textlength,xticks,yticks,zticks = self.textlength,self.xticks,self.yticks,self.zticks
        base_left_offset = 0.03
        ax_base_left_offset =0.08
        top_up_offset = -0.025
        bottom_up_offset = 0.05
        ax_top_up_offset = -0.075
        ax_bottom_up_offset = 0.1
        textoffset = 0.025
        ticksoffset = 1/float(self.num_ticks-1)
        if abs(N.dot([1,0,0],self.view)) < abs(N.dot([0,1,0],self.view)):
            point = N.array([self.sides[0]-1.5,3.5-self.sides[1],0])
            line = N.array([0,0,1])
            if abs(self.new_x[0]) > abs(self.new_x[1]) :
                if point[0]*self.new_x[0] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[1]*self.new_x[1] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[0]) > abs(self.new_y[1]) :
                if point[0]*self.new_y[0] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            else :
                if point[1]*self.new_y[1] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            if (abs(N.dot(line,self.new_x)) > 0.4) and(abs(N.dot(line,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line,self.new_y)))
            offset = left_offset*factor*self.new_x + up_offset*factor2*self.new_y
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
    
            for a in range(self.num_ticks):
                self.PutText3d(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0)-offset[0],3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)-offset[1],-0.5 + a*ticksoffset-offset[2],yticks[a])        
            self.PutText3d(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0)-ax_offset[0],3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)-ax_offset[1],0-ax_offset[2],'Y')

            glBegin(GL_LINES)
            glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),-0.5-self.offset)
            glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),0.5+self.offset)
            for a in range(self.num_ticks):
                glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)+0.025*cmp(3.5-self.sides[1],0),-0.5+a*ticksoffset)
                glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),-0.5+a*ticksoffset)
            
        else :
            point = N.array([1.5-self.sides[0],self.sides[1]-3.5,0])
            if abs(self.new_x[0]) > abs(self.new_x[1]) :
                if point[0]*self.new_x[0] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[1]*self.new_x[1] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[0]) > abs(self.new_y[1]) :
                if point[0]*self.new_y[0] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset = ax_bottom_up_offset
            else :
                if point[1]*self.new_y[1] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            line = N.array([0,0,1])
            if (abs(N.dot(line,self.new_x)) > 0.4) and (abs(N.dot(line,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line,self.new_y)))
            offset = left_offset*factor*self.new_x
            offset = offset + up_offset*factor2*self.new_y
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
            for a in range(self.num_ticks) :
                self.PutText3d(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)-offset[0],self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0)-offset[1],-0.5+a*ticksoffset-offset[2],yticks[a])
            self.PutText3d(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)-ax_offset[0],self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0)-ax_offset[1],0-ax_offset[2],'Y')

            glBegin(GL_LINES)
            glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),-0.5-self.offset)
            glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),0.5+self.offset)
            for a in range(self.num_ticks):
                glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)+0.025*cmp(1.5-self.sides[0],0),self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),-0.5+ticksoffset*a)
                glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),-0.5+ticksoffset*a)
        glEnd()

         #-------------------------------------------------
        if abs(N.dot([1,0,0],self.view)) < abs(N.dot([0,0,1],self.view)):
            glBegin(GL_LINES) 
            glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),-0.5-self.offset,5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),0.5+self.offset,5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            for a in range(self.num_ticks):
                glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),-0.5+ticksoffset*a,5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)+0.025*cmp(5.5-self.sides[2],0))
                glVertex3f(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0),-0.5+ticksoffset*a,5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            glEnd() 
            point = N.array([self.sides[0]-1.5,0,5.5-self.sides[2]])
            if abs(self.new_x[0]) > abs(self.new_x[2]) :
                if point[0]*self.new_x[0] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[2]*self.new_x[2] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[0]) > abs(self.new_y[2]) :
                if point[0]*self.new_y[0] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            else :
                if point[2]*self.new_y[2] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            line = N.array([0,1,0])
            if (abs(N.dot(line ,self.new_x)) > 0.4) and (abs(N.dot(line ,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line ,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line ,self.new_y)))
            offset = left_offset*factor*self.new_x
            offset = offset + up_offset*factor2*self.new_y 
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
            for a in range(self.num_ticks):
                self.PutText3d(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0)-offset[0],-0.5+ticksoffset*a-offset[1],5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)-offset[2],zticks[a])          
            self.PutText3d(self.sides[0]-1.5+self.offset*cmp(self.sides[0]-1.5,0)-ax_offset[0],0-ax_offset[1],5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)-ax_offset[2],'Z')

        else :
            glBegin(GL_LINES) 
            glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),-0.5-self.offset,self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),0.5+self.offset,self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            for a in range(self.num_ticks):
                glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)+0.025*cmp(1.5-self.sides[0],0),-0.5+ticksoffset*a,self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
                glVertex3f(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0),-0.5+ticksoffset*a,self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            glEnd()
            point = N.array([1.5-self.sides[0],0,self.sides[2]-5.5])
            if abs(self.new_x[0]) > abs(self.new_x[2]) :
                if point[0]*self.new_x[0] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[2]*self.new_x[2] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[0]) > abs(self.new_y[2]) :
                if point[0]*self.new_y[0] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            else :
                if point[2]*self.new_y[2] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            line = N.array([0,1,0])
            if (abs(N.dot(line,self.new_x)) > 0.4) and (abs(N.dot(line,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line,self.new_y)))
            offset = left_offset*factor*self.new_x
            offset = offset + up_offset*factor2*self.new_y
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
            for a in range(self.num_ticks):
                self.PutText3d(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)-offset[0],-0.5+ticksoffset*a-offset[1],self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0)-offset[2],zticks[a])
            self.PutText3d(1.5-self.sides[0]+self.offset*cmp(1.5-self.sides[0],0)-ax_offset[0],0-ax_offset[1],self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0)-ax_offset[2],'Z')

        #------------------------------------------------------------
        if abs(N.dot([0,1,0],self.view)) < abs(N.dot([0,0,1],self.view)):
            glBegin(GL_LINES) 
            glVertex3f(-0.5-self.offset,self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            glVertex3f(0.5+self.offset,self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            for a in range(self.num_ticks):
                glVertex3f(-0.5+ticksoffset*a,self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)+0.025*cmp(5.5-self.sides[2],0))
                glVertex3f(-0.5+ticksoffset*a,self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0),5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0))
            glEnd()
            point = N.array([0,self.sides[1]-3.5,5.5-self.sides[2]])
            if abs(self.new_x[1]) > abs(self.new_x[2]) :
                if point[1]*self.new_x[1] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[2]*self.new_x[2] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[1]) > abs(self.new_y[2]) :
                if point[1]*self.new_y[1] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            else :
                if point[2]*self.new_y[2] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            line = N.array([1,0,0])
            if (abs(N.dot(line,self.new_x)) > 0.4) and (abs(N.dot(line,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line,self.new_y)))
            offset = left_offset*factor*self.new_x
            offset = offset + up_offset*factor2*self.new_y
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
            for a in range(self.num_ticks):
                self.PutText3d(-0.5+ticksoffset*a-offset[0],self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0)-offset[1],5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)-offset[2],xticks[a])
            self.PutText3d(0-ax_offset[0],self.sides[1]-3.5+self.offset*cmp(self.sides[1]-3.5,0)-ax_offset[1],5.5-self.sides[2]+self.offset*cmp(5.5-self.sides[2],0)-ax_offset[2],'X')

        else :
            glBegin(GL_LINES) 
            glVertex3f(-0.5-self.offset,3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            glVertex3f(0.5+self.offset,3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            for a in range(self.num_ticks):
                glVertex3f(-0.5+ticksoffset*a,3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)+0.025*cmp(3.5-self.sides[1],0),self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
                glVertex3f(-0.5+ticksoffset*a,3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0),self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0))
            glEnd()
            point = N.array([0,3.5-self.sides[1],self.sides[2]-5.5])
            if abs(self.new_x[1]) > abs(self.new_x[2]) :
                if point[1]*self.new_x[1] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            else :
                if point[2]*self.new_x[2] > 0 :
                    #right
                    left_offset = -base_left_offset
                    ax_left_offset = -ax_base_left_offset-textlength*textoffset
                else :
                    #left
                    left_offset = base_left_offset +textlength*textoffset
                    ax_left_offset = ax_base_left_offset+(textlength+1)*textoffset
            if abs(self.new_y[1]) > abs(self.new_y[2]) :
                if point[1]*self.new_y[1] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            else :
                if point[2]*self.new_y[2] > 0 :
                    #top
                    up_offset = top_up_offset
                    ax_up_offset  = ax_top_up_offset
                else :
                    #bottom
                    up_offset = bottom_up_offset
                    ax_up_offset  = ax_bottom_up_offset
            line = N.array([1,0,0])
            if (abs(N.dot(line,self.new_x)) > 0.4) and (abs(N.dot(line,self.new_y)) > 0.21) :
                factor = 1
            else :
                factor = 1.2*(1-abs(N.dot(line,self.new_x)))
            factor2 = 1.2*(1-abs(N.dot(line,self.new_y)))
            offset = left_offset*factor*self.new_x
            offset = offset + up_offset*factor2*self.new_y
            ax_offset = ax_left_offset*factor*self.new_x+ax_up_offset*factor2*self.new_y
            for a in range(self.num_ticks):
                self.PutText3d(-0.5+ticksoffset*a-offset[0],3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)-offset[1],self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0)-offset[2],xticks[a])       
            self.PutText3d(0-ax_offset[0],3.5-self.sides[1]+self.offset*cmp(3.5-self.sides[1],0)-ax_offset[1],self.sides[2]-5.5+self.offset*cmp(self.sides[2]-5.5,0)-ax_offset[2],'X')
       
        glLineWidth(self.linewidth)
        glColor3f(0.0,0.0,0.0)
        glEnable(GL_LIGHTING)
        
    def redraw(self,o):
        self.InitGfx()
        self.DrawPlotBox()
        self.DrawPlotBoxLines()
        
        glLineWidth(self.linewidth)
        self.InsertAxesText()
        self.drawplot()
        glLineWidth(0.1)

    def save(self):
        myFormats = [
            ('Windows Bitmap','*.bmp'),
            ('Enhanced Windows Metafile','*.emf'),
            ('Encapsulated PostScript','*.eps'),
            ('CompuServe GIF','*.gif'),
            ('JPEG / JPG','*.jpg'),
            ('Zsoft Paintbrush','*.pcx'),
            ('Portable Network Graphics','*.png'),
            ('Portable Pixelmap','*.ppm'),
            ('Tagged Image File Format','*.tif'),
            ]
        fileName = tkFileDialog.asksaveasfilename(filetypes=myFormats ,title="Save the image as...")
        if fileName :
            self.SaveTo(fileName)
	
    def SaveTo(self,filename):
	"""Save current buffer to filename in format"""
	import Image # get PIL's functionality...
	viewport = glGetIntegerv(GL_VIEWPORT)
	width, height = viewport[2],viewport[3]
	glPixelStorei(GL_PACK_ALIGNMENT, 1)
	data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
	image = Image.fromstring( "RGB", (width, height), data )
	image = image.transpose( Image.FLIP_TOP_BOTTOM)
	image.save( filename)
	print 'Saved image to %s'% (os.path.abspath( filename))
	return image
            
    def make_plot(self,colors = Colors):
        """Draw a plot in a Tk OpenGL render window."""
        
        # Some api in the chain is translating the keystrokes to this octal string
        # so instead of saying: ESCAPE = 27, we use the following.
        ESCAPE = '\033'

        # Number of the glut window.
        window = 0
        
        # Create render window (tk)
        f = Frame()
  	f.pack(side = 'top')
  	self.offset = 0.05
  	self.textlength,self.xticks,self.yticks,self.zticks =self.GetFormatMaxTextLength() #todo: each tick own offset
  	o = Opengl(width = 640, height = 480, double = 1, depth = 1)
  	o.redraw = self.redraw
  	quit = Button(f, text = 'Quit', command = sys.exit)
  	quit.pack({'side':'top', 'side':'left'})
  	help = Button(f, text = 'Help', command = o.help)
  	help.pack({'side':'top', 'side':'left'})
  	save = Button(f, text = 'Save', command = self.save)
  	save.pack({'side':'top', 'side':'left'})
  	reset = Button(f, text = 'Reset', command = o.reset)
  	reset.pack({'side':'top', 'side':'left'})
  	o.pack(side = 'top', expand = 1, fill = 'both')
  	o.set_background(1,1,1)
        o.set_centerpoint(0.0, 0.0, 0.0)
        o.set_eyepoint(3)
        o.autospin = 1
  	o.mainloop()


class mesh(figure) :
    """class for surface meshes"""
    def __init__(self,X,Y,Z,colors = Colors,linewidth = 0.1,fill_mesh = 0,precision=3,num_ticks=5):
        self.Colors = colors
        self.linewidth= linewidth
        self.fill_mesh = fill_mesh
        self.precision = precision
        self.num_ticks = num_ticks
        self.Rescale(X,Y,Z)
        self.make_plot(colors)
        return

    def drawplot(self) :
        glDisable(GL_LIGHTING)

        X_shift = self.X-0.5
        Y_shift = self.Y-0.5
        Z_shift = self.Z-0.5
        
        plotsizex,plotsizey = N.shape(self.X)

        if self.fill_mesh :
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	    glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0, 1.0);
            glColor3f(0.98, 0.98, 0.98)
            glBegin(GL_TRIANGLES)
            for i in range(plotsizex-1):
                for j in range(plotsizey-1):
                    self.SetColor(self.Z[i,j])
                    glVertex3f( X_shift[i,j],Z_shift[i,j],Y_shift[i,j])
                    self.SetColor(self.Z[i,j])
                    glVertex3f( X_shift[i+1,j],Z_shift[i+1,j],Y_shift[i+1,j])
                    self.SetColor(self.Z[i,j])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1] )
                    self.SetColor(self.Z[i+1,j])
                    glVertex3f( X_shift[i+1,j], Z_shift[i+1,j],Y_shift[i+1,j] )
                    self.SetColor(self.Z[i,j+1])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1])
                    self.SetColor(self.Z[i+1,j+1])
                    glVertex3f( X_shift[i+1,j+1], Z_shift[i+1,j+1],Y_shift[i+1,j+1] )
            glEnd()
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glColor3f(0.0, 0.0, 0.0)
            glBegin(GL_TRIANGLES)
            for i in range(plotsizex-1):
                for j in range(plotsizey-1):
                    glVertex3f( X_shift[i,j],Z_shift[i,j],Y_shift[i,j])
                    glVertex3f( X_shift[i+1,j],Z_shift[i+1,j],Y_shift[i+1,j])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1] )
                    glVertex3f( X_shift[i+1,j], Z_shift[i+1,j],Y_shift[i+1,j] )
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1])
                    glVertex3f( X_shift[i+1,j+1], Z_shift[i+1,j+1],Y_shift[i+1,j+1] )
            glEnd()
        else :
            #Simple hidden surface removal
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glColor3f(0.0, 0.0, 0.0)
            glBegin(GL_TRIANGLES)
            for i in range(plotsizex-1):
                for j in range(plotsizey-1):
                    self.SetColor(self.Z[i,j])
                    glVertex3f( X_shift[i,j],Z_shift[i,j],Y_shift[i,j])
                    self.SetColor(self.Z[i+1,j])
                    glVertex3f( X_shift[i+1,j],Z_shift[i+1,j],Y_shift[i+1,j])
                    self.SetColor(self.Z[i,j+1])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1] )
                    self.SetColor(self.Z[i+1,j])
                    glVertex3f( X_shift[i+1,j], Z_shift[i+1,j],Y_shift[i+1,j] )
                    self.SetColor(self.Z[i,j+1])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1])
                    self.SetColor(self.Z[i+1,j+1])
                    glVertex3f( X_shift[i+1,j+1], Z_shift[i+1,j+1],Y_shift[i+1,j+1] )
            glEnd()

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0, 1.0);
            glColor3f(0.98, 0.98, 0.98)
            glBegin(GL_TRIANGLES)
            for i in range(plotsizex-1):
                for j in range(plotsizey-1):
                    glVertex3f( X_shift[i,j],Z_shift[i,j],Y_shift[i,j])
                    glVertex3f( X_shift[i+1,j],Z_shift[i+1,j],Y_shift[i+1,j])
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1] )
                    glVertex3f( X_shift[i+1,j], Z_shift[i+1,j],Y_shift[i+1,j] )
                    glVertex3f( X_shift[i,j+1], Z_shift[i,j+1],Y_shift[i,j+1])
                    glVertex3f( X_shift[i+1,j+1], Z_shift[i+1,j+1],Y_shift[i+1,j+1] )
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL);

        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_LIGHTING)

class plot3_points(figure) :
    """ class for plotting points in 3d"""
    def __init__(self,Points,colors = Colors,linewidth = 0.1,pointsize=5,precision=3,num_ticks=5):
        self.Colors = colors
        self.linewidth= linewidth
        self.pointsize = pointsize
        self.precision = precision
        self.num_ticks = num_ticks
        self.Rescale(Points)
        self.make_plot(colors)
        return

    def Rescale(self,Points) :
	self.xmin = min(Points[:,0])
        self.xmax = max(Points[:,0])
        self.ymin = min(Points[:,1])
        self.ymax = max(Points[:,1])
        self.zmin = min(Points[:,2])
        self.zmax = max(Points[:,2])

	#rescaling
        self.Points = N.array(Points)
        self.Points[:,0] = (self.Points[:,0]-self.xmin)/abs(self.xmax-self.xmin)
	self.Points[:,1] = (self.Points[:,1]-self.ymin)/abs(self.ymax-self.ymin)
	self.Points[:,2] = (self.Points[:,2]-self.zmin)/abs(self.zmax-self.zmin)

    def drawplot(self) :
        Points = self.Points - 0.5
	
        glDisable(GL_LIGHTING)
        glColor3f(0,0,1.0)
	glPointSize(self.pointsize)
	glBegin(GL_POINTS)
	for i in range(Points.shape[0]) :
                self.SetColor(Points[i,2])
		glVertex3f(Points[i,0],Points[i,2],Points[i,1])
	glEnd()
        glEnable(GL_LIGHTING)
    
class plot3(figure) :
    """ similar to matlab plot3 functionality"""
    def __init__(self,X,Y,Z,colors = Colors,linewidth = 0.1,precision=3,num_ticks=5):
        self.Colors = colors
        self.linewidth= linewidth
        self.precision = precision
        self.num_ticks = num_ticks
        self.Rescale(X,Y,Z)
        self.make_plot(colors)
        return
    
    def drawplot(self) :
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
        glDisable(GL_LIGHTING)
            
        glBegin(GL_LINES)

        glColor3f(self.Colors[0,0], self.Colors[0,1], self.Colors[0,2])

        X_shift = self.X-0.5
        Y_shift = self.Y-0.5
        Z_shift = self.Z-0.5
    
        plotsize = N.size(self.X)
        plot_range = range(plotsize-1)

        if len(self.X.shape) == 1 :
            for i in plot_range:
                glVertex3f( X_shift[i],Z_shift[i],Y_shift[i])
                glVertex3f( X_shift[i+1],Z_shift[i+1],Y_shift[i+1])        

        else :
            num_lines = self.X.shape[0]
            num_points = self.X.shape[1]
            for i in range(num_lines):
                self.SetColor(i/(float)(num_lines))
                for j in range(num_points-1):
                    glVertex3f( X_shift[i,j],Z_shift[i,j],Y_shift[i,j])
                    glVertex3f( X_shift[i,j+1],Z_shift[i,j+1],Y_shift[i,j+1])   

        glEnd()
        glEnable(GL_LIGHTING)

class plotrbf(figure,mesh) :
    """Class written to visualise some radial basis function interpolations """
    def __init__(self,X,Y,Z,Points,colors = Colors,linewidth=0.1,fill_mesh = 0,pointsize=5,precision=3,num_ticks=5):
        self.Colors = colors
        self.linewidth= linewidth
        self.fill_mesh = fill_mesh
        self.pointsize = pointsize
        self.precision = precision
        self.num_ticks = num_ticks
	self.Rescale(X,Y,Z,Points)
        self.make_plot(colors)
        return

    def Rescale(self,X,Y,Z,Points) :
        figure.Rescale(self,X,Y,Z)
        self.Points = N.array(Points)
	#rescaling
	self.Points[:,0] = (self.Points[:,0]-self.xmin)/abs(self.xmax-self.xmin)
	self.Points[:,1] = (self.Points[:,1]-self.ymin)/abs(self.ymax-self.ymin)
	self.Points[:,2] = (self.Points[:,2]-self.zmin)/abs(self.zmax-self.zmin)

        
    def drawplot(self) :
        mesh.drawplot(self)

	#rescaling
	Points = self.Points - 0.5
	
        glDisable(GL_LIGHTING)
        glColor3f(0,0,0)
	glPointSize(self.pointsize)
	glBegin(GL_POINTS)
	for i in range(len(Points[0])) :
		glVertex3f(Points[i,0],Points[i,2],Points[i,1])
	glEnd()
        glEnable(GL_LIGHTING)

        
