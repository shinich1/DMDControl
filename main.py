"""
sequencer.
here you can define the sequence, such as triggering, delay etc.
dmdmain_usb is the sequencer when using usb upload.
inheriting from inst.py, zernike.py, hologram.py and dmd.py
"""
import numpy as np
import time
import os
import sys
import threading
import pygame
from inst import Daq#,oscillo
from zernike import zernikeoptim
from hologram import hologram
from dmd import disp
from datetime import datetime as dt
import matplotlib.pyplot as plt
import scipy.misc as scm

class dmdmain(disp,hologram,Daq,zernikeoptim):
    def __init__(self):
        print "initializing. it may take seconds"
        self.stop_trig=False
        zernikeoptim.__init__(self)
        Daq.__init__(self,["Dev1/ai2","Dev1/ai1"])
        #oscillo.__init__(self)
        hologram.__init__(self)
        self.phasemap = self.calcmaps()
        self.path="../data/"
        self.pathcheck()
        self.defaultcoefs = np.genfromtxt("../data/default/coefs.csv",delimiter=",")
        print "system initialized! starting pygame.."
        disp.__init__(self)




    """

    note for self.display(image,width,P,alpha,zernikeoptim=True,yb=False,ybscale=1)
        image : "gaus","tem","sinc","linsinc",path (for a 1080*1080 image.)
        width : w0=width*2.8mm on fourier plane (before objective)
                    expected width on image(atom) plane:  w~=2*f*lambda/w0 =794nm/width

        P: binarization criteria. default is 1
        alpha: binarization diffusion coefficient. 5~20, default is 8
        (optional) zernikeoptim: whether or not to use zernike coefs
        (optional) yb: "Yb" letters using 17 gaus beams
        (optional) yb scale: scale of yb letters.
    """



    
    def manualoptim(self):
        """
        manually change zernike coefs through LAN
        please use text editor, not microsoft excel
        """
        self.calcmaps()
        for i in xrange(1000):
            self.coefs=self.defaultcoefs[:,0]
            self.display("gaus",width=2.0,P=1,alpha=8.)
            self.calcmaps()
            self.getinput()#abort with esc+z
        self.stop()


    def zernikeoptimize(self):
        """
        scan zernike coefficients
        can change coefs through LAN during scan
        """
        self.calcmaps()
        for i in xrange(15):
            self.coefs=self.defaultcoefs[:,0]+self.defaultcoefs[:,1]*i
            self.display("gaus",width=2.0,P=1,alpha=8.)
            self.calcmaps()
            #self.waittrig()
            self.getinput()#abort with esc+z
        self.stop()


    def coefoptimize(self):
        """
        scan alpha, P, width(of gaussian)
        """
        self.calcmaps()
        for i in xrange(20): #starts from 0
            self.display("gaus",width=2.0,P=1,alpha=i+1.)   #alpha
            #self.display("gaus",width=2.0,P=0.8+i*0.2,alpha=8.)   #P
            #self.display("gaus",width=0.3+i*0.05,P=1,alpha=8.)   #width
            #self.waittrig()
            self.getinput()
        self.stop()

    def dispyb(self):
        self.calcmaps()
        for i in xrange(5):
            self.display("gaus",width=1.0,P=1,alpha=8.,zernikeoptim=True,yb=True,ybscale=1+i*3)
            #self.waittrig()
            self.getinput()
        self.stop()





    #####################################################

    #####################################################

    def display(self,img,width=2.0,P=1.,alpha=8.,zernikeoptim=True,yb=False,ybscale=1):
        Xshift=0
        Yshift=0
        correction=True
        self.fimage=self.imgfft(width,img)
        posX=np.array([0])
        posY=np.array([0])

        if yb:
            posX,posY=self.yb()
            posX*=ybscale
            posY*=ybscale

        self.hologramize(self.phasemap,Xshift,Yshift,correction,posX,posY,P,alpha)
        #pygame.display.flip()
        self.swap_event.set()
        f=open(self.path+"/log.txt","a")
        f.write( dt.now().strftime('%H %M') + " img= "+img +" coefs:\n" + str(self.coefs) + "\n")
        f.close()   


    def pathcheck(self):
        #once a day.
        self.path += dt.now().strftime('%m%d')
        if not os.path.exists(self.path):
            os.makedirs(self.path)    
        #log file
        if not os.path.exists(self.path+"/log.txt"):
            f=open(self.path+"/log.txt","a")
            f.write("log file for " + dt.now().strftime('%m %d') + " starting at " + dt.now().strftime('%H:%M'))
            f.close()



if __name__ == '__main__':
    a = dmdmain()
    #a.manualoptim()
    #a.zernikeoptimize()
    #a.coefoptimize()
    a.dispyb()

