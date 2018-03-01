"""
zernike optimization GUI and program.

22/1/2017 author: sunami
"""


import os
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class zernikeoptim():
    def __init__(self):
        
        #a thread for reading csv once in 2 seconds
        self.csvupdate = threading.Thread(target = self.csvupdate)
        self.csvupdate.start()

        """
        common aberration components:
        0:tip
        1:tilt
        2:defocus
        3:spherical
        4:comav
        5:comah
        6:astigv
        7:astigh
        8:trefoilv
        9:trefoilh
        10:second spherical
        11:glass plate compensation
        """
        self.defaultcoefs = np.genfromtxt("./coefs.csv",delimiter=",")
        self.coefs = np.genfromtxt("./coefs.csv",delimiter=",")[:,0]
        self.zernikemaps=zernikemaps() #10*1080*1080 numpy array. valid only in circular region.

    def calcmaps(self): #coefs:1*8 array
        phase = np.zeros((1080,1080))
        for i in xrange(12):
            phase += self.coefs[i]*self.zernikemaps[i]
        self.phasemap = phase
        return phase

    def csvupdate(self):
        while not self.stop_trig:
            time.sleep(1)
            self.defaultcoefs = np.genfromtxt("./coefs.csv",delimiter=",")


def zernikemaps():
    """
    define phasemap for each zernike components.
    scaled to max=1 in circular region.
    """
    size = 1080
    X,Y = np.meshgrid(np.linspace(-1,1,size),np.linspace(-1,1,size))
    r=np.sqrt(X**2+Y**2)
    r2=r+0.0001*(r==0) #prevent deviding with 0 at 0,0
    co=np.divide(X,r2)
    si=np.divide(Y,r2)
    si2=2*si*co
    co2=1-2*si**2
    si3=3*si-4*(si**3)
    co3=4*(co**3)-3*co
    phasemap = np.zeros((12,size,size))
    phasemap[0] = X
    phasemap[1] = Y
    phasemap[2] = 2*r**2-1  #defocus
    phasemap[3] = 6*r**4-6*r**2+1 #spherical
    phasemap[4] =(3*r**3-2*r)*si #comav
    phasemap[5] = (3*r**3-2*r)*co #comah
    phasemap[6] = r**2*si2 #astigv
    phasemap[7] = r**2*co2 #astigh
    phasemap[8] = (r**3)*co3 #trev
    phasemap[9] = (r**3)*si3 #treh
    phasemap[10] = (20*r**6 - 30*r**4 + 12*r**2 -1)  # secondary spherical
    phasemap[11] = 1.36*(r*0.5)**4 +(r*0.5)**6 #glass thickness error

    return phasemap






if __name__ == '__main__':

    a=zernikemaps()
    plt.figure()
    plt.imshow(a[0])
    plt.show()
    







    """
    x,y=np.meshgrid(np.arange(-540,540),np.arange(-540,540))
    r=1*(x**2+y**2<580**2)
    a=zernikeoptim()
    
    plt.figure(1,(27,10))
    for i in xrange(12):
        plt.subplot(6,2,i+1)
        phase = a.coefs[i]*a.zernikemaps[i]*r
        plt.imshow(phase)
        plt.colorbar()
    
    phase = a.coefs[10]*a.zernikemaps[10]*r
    phase += a.coefs[11]*a.zernikemaps[11]*r
    plt.figure()
    plt.imshow(phase)
    plt.colorbar()
    plt.figure(2)
    a.calcmaps()
    plt.imshow(a.phasemap*r-a.coefs[0]*a.zernikemaps[0]*r-a.coefs[1]*a.zernikemaps[1]*r)
    plt.colorbar()
    plt.show()
    """