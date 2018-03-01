"""
hologram.py
takes the phasemap and target image (or function)
returns the hologram
"""
import numpy as np
import numexpr as ne
from PIL import Image as pil
import pygame
import matplotlib.pyplot as plt
import scipy.special as special


class hologram():
    def __init__(self):
        self.X,self.Y = np.meshgrid(np.arange(1080)-540,np.arange(1080)-540)
        self.rand = np.random.rand(1080,1080)
        self.fimage = self.imgfft(1.2,"gaus")
        self.amplitudemap = np.ones((1080,1080))

    def hologramize(self,phm,Xshift,Yshift,correction,posX,posY,P=1,alpha=14.):
        iteration = posX.shape[0]
        dmdpattern = np.zeros((1080,1920),dtype=np.int)
        fimage1 = self.fimage[1]
        gratingphase = np.zeros(self.X.shape)
        omega = ne.evaluate("fi/ph",local_dict={'fi':self.fimage[0],'ph':self.amplitudemap})
        omega = ne.evaluate("omega/maximum",local_dict={'omega':omega,'maximum':np.amax(omega)})
        for i in xrange(iteration):
            if (correction == True):
                phasesum = ne.evaluate("((-phasemap1+fimage1+X*xshift+Y*yshift)%6.283185)*12./6.283185",local_dict={'X':self.X,'Y':self.Y,'phasemap1':phm,'fimage1':fimage1,'xshift':posX[i]+Xshift,'yshift':posY[i]+Yshift})
            elif (correction ==False):
                phasesum = ne.evaluate("((fimage1+X*xshift+Y*yshift)%6.283185)*12./6.283185",local_dict={'X':self.X,'Y':self.Y,'phasemap1':phm,'fimage1':fimage1,'xshift':posX[i]+Xshift,'yshift':posY[i]+Yshift})
            else:
                print "error"
            gratingphase += ne.evaluate("abs((Y-X+phasesum)%12.-6)/6.",local_dict={'X':self.X,'Y':self.Y,'phasesum':phasesum})
        value =ne.evaluate("rand*2/(tanh(alpha*(gratingphase+omega/2))-tanh(alpha*(gratingphase-omega/2)))",local_dict={'gratingphase':gratingphase/iteration,'omega':omega,'alpha':alpha,'rand':self.rand})
        
        #hologram inside circular area
        dmdpattern[0:1080,420:1500]=ne.evaluate("1*(value <P)*(X**2+Y**2<620**2)",local_dict={'P':P,'value':value,'X':self.X,'Y':self.Y})
        
        #return dmdpattern
        self.image2 = pygame.surfarray.make_surface(dmdpattern.T*255)     

    def imgfft(self,width=0,image="gaus"):
        fimage = np.zeros((3,self.X.shape[0],self.X.shape[1]),dtype=np.float64)
        if (image == "gaus"):
            fimage[0] = self.gaus(width) #amplitude only
        elif (image == "tem"):
            fimage[0] = np.absolute(self.tem(width))
            fimage[1] = math.pi*(self.tem(width)<0)
        elif (image =="sinc"):
            fimage[0] = np.absolute(self.sinc(width))**2
            fimage[1] = math.pi*(self.sinc(width)<0)
        elif (image =="linsinc"):
            fimage[0] = np.absolute(self.linsinc(width))
            fimage[1] = math.pi*(self.linsinc(width)<0)
        elif (image =="lgaus"):
            print "lgaus"
            fimage[0] = np.absolute(self.lgaus(width,1,1))
            fimage[1] = np.angle(self.lgaus(width,1,1))
        elif (image == "circles"):
            fimage[0] = 1*(self.X**2+self.Y**2 <100)+((self.X-100)**2+self.Y**2 <100)
        elif (image=="confine"):
            #confiment potential which compensates for large gaussian profile of optical lattice
            fimage[0] = np.absolute(-1*self.gaus(0.7)+self.lgaus(width,0,1))
            fimage[1] = np.angle(self.lgaus(self.X,self.Y,width,0,1))
            fimage[1,::2,1::2] = fimage[1,::2,::2]
            fimage[1,1::2,::2] = fimage[1,::2,::2]
            fimage[1,1::2,1::2] = fimage[1,::2,::2]
        else:
            #for arbitrary image to be fft'ed. image shoule be large enough
            im =np.array(pil.open(image))[:,:,1]/255. 
            fftim = np.fft.fftshift(np.fft.fft2(im))
            fimage[0] = np.absolute(fftim)
            fimage[1] = np.angle(fftim)
            fimage[1,::2,1::2] = fimage[1,::2,::2]
            fimage[1,1::2,::2] = fimage[1,::2,::2]
            fimage[1,1::2,1::2] = fimage[1,::2,::2]
            
        return fimage



    def imload(self,path):
        return np.array(pil.open(path))[:,:,1]/255.

    def gaus(self,width):
        return 1*np.exp(-(self.X**2+self.Y**2)/(2.*(width*self.X.shape[0]*0.1*2)**2))

    def tem(self,width,order1=1,order2=1):
        order = np.zeros((4,4))
        order[order1,order2] =1
        width *= self.X.shape[0]/8
        return 1*np.polynomial.hermite.hermval2d(np.sqrt(2)*self.X/width,np.sqrt(2)*self.Y/width,order)*np.exp(-(self.X**2+self.Y**2)/width**2)

    def lgaus(self,width,order=2,lorder=0):
        width *= 0.3
        size = self.X.shape[0]
        x = np.linspace(0,10,size)-5
        self.X,self.Y=np.meshgrid(x,x)
        r=np.sqrt(self.X**2+self.Y**2)
        laguerre = special.genlaguerre(order,lorder)
        ph=np.arctan(self.Y/self.X)+3.1415*(self.X-self<0)
        return 1*(r**lorder)*laguerre(2*r**2/width**2)*np.exp(-r**2/width**2)*np.exp(-1j*(r**2)/3)*np.exp(1j*lorder*ph) #replace 1 by phi

    def yb(self):

        yshift = -1*np.array([-0.04,-0.02,0.02 ,0.04 , 0    ,0    , 0 ,-0.06 ,-0.06 ,-0.06 ,-0.06  ,-0.06 ,-0.08 ,-0.08 ,-0.1  ,-0.09 ,-0.09])
        xshift = -1*np.array([0.04 ,0.02 ,0.02 ,0.04 ,-0.02 ,-0.04, 0 ,0     ,0.02  ,0.04  ,-0.025 ,-0.05 ,-0.00 ,-0.05 ,-0.027 ,-0.02 ,-0.04])
        return xshift,yshift

    def sinc(self,width):
        width *= self.X.shape[0]/5.
        return np.sinc(np.sqrt(self.X**2+self.Y**2)/width)

    def linsinc(self,width):
        width *= self.X.shape[0]/1.7
        return np.sinc((self.X-self.Y)/width)




if __name__ == '__main__':
    a=hologram()
    a.imgfft(1,"lgaus")
    plt.figure()
    plt.imshow(a.fimage[0])
    plt.show()
