
import time, threading
import pygame, sys
from struct import *
from pygame.locals import *
#import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.filters import threshold_adaptive
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import numexpr as ne
import cProfile
cimport cython
cimport numpy as np
ctypedef np.int_t DTYPEi_t
ctypedef np.float64_t DTYPE_t
#cimport cython
#np.import_array()


"""
todo
1. interface andor
2. triggering for image acquisition = andor.py has waitforaquisition()!!!!. 
2. triggering for beam generation(which way is the fastest? shutter? shutter on & hologram on?)
3. zernike coefs
4. lattice positioin calibration?? -> deconvolution with known knowledge
5. cythonize hologram.py?
6. 
"""

class feedback():
    def __init__(self):
        """
        init daq, andor, pygame
        run prepareholograms()
        """

        self.X,self.Y = np.meshgrid(np.arange(1080)-540,np.arange(1080)-540)
        self.rand = np.random.rand(1080,1080)
        self.fimage = self.imgfft(1.2,"gaus")
        self.amplitudemap = np.ones((1080,1080))

        pygame.init()
        self.windowSurface = pygame.display.set_mode((1920, 1080), 0,8)#   FULLSCREEN)#0, 8)
        self.image = pygame.Surface((1920,1080)) 
        self.image2 = pygame.Surface((1920,1080)) #pygame.image.load("../data/default/hologram2.jpg").convert() 

        pygame.event.set_grab(True)
        pygame.mouse.set_visible(0)
        self.stop_event = threading.Event() #flag stop or not 
        self.swap_event = threading.Event()  #flag increment or not
        self.count=0
        self.count2=0

        ##lattice site position calibration with tilt/tip
        self.latticex=0.1
        self.latticey=0.2
        self.latticex0=1.1
        self.latticey0=3.0

        #phasemap
        self.phm=np.zeros((1080,1080))

        #prepare holograms for 10*10 lattice sites
        self.preholograms=np.zeros((10,10,1080,1080))
        self.prepareholograms(10,10)
        #create thread and start
        self.thread = threading.Thread(target = self.showing)
        self.thread.start()
        #self.windowSurface.blit(self.image2,(0,0))
        #pygame.display.flip()

        #pygame.surfarray.blit_array()
        '''you can blit array directly fast. values must be [0,255] and 8 depth'''


    #############################################
    """ sequence, pygame and andor functions """
    #############################################
    def sequence(self):
        self.prepareholograms()
        for i in xrange(1000):
            positionx,positiony = self.waitforimage()
            self.superpose(positionx,positiony)
            self.beamon(dulation=0.4)
            self.savedata()


    def showing(self):
    # run the pygame loop
        #pygame.mouse.set_visible(False)
        while not self.stop_event.is_set():
            self.count +=1

            if self.swap_event.is_set():
                #self.image = self.image2
                self.windowSurface.blit(self.image2,(0,0))#,(50,0,1000,1000))
                pygame.display.flip()
                self.swap_event.clear()
            #for event in pygame.event.get(KEYDOWN):
            #    if event.key == K_z:
            #        print "press"
            #        self.stop()    
        self.stop_trig = True


    def stop(self):
        """stop thread"""
        self.stop_event.set()
        pygame.quit()
        self.thread.join() 
        self.stop_trig=True
        print "finished. roop count: " + str(self.count) + " , " + str(self.count2)

    def beamon(self, dulation=0.4):
        """
        calculate the hologram, display and then 
        daq digital on -> refresh stop + DMD AOM or shutter
        after dulation, turn daq digital off (and display darkim)
        """
        pass

    def timetest(self):
        xlist=np.array([0,1,2,3,4,5,6,7,8,9])
        ylist=np.array([0,1,2,3,4,5,6,7,8,9])
        cProfile.run("self1.superpose(xlist,ylist)")
        self.stop()

    def waitforimage(self,deconvolution=False):
        """
        wait for andor image and compute deconvolution or peak detection.
        timeout=30s?
        """
        self.data=getimage()
        if deconvolution:
            pass
        else:
            filt = gaussian_filter(data,1)
            thres = threshold_adaptive(filt, 13, offset=-150)
            er = morph.erosion(thres,self.disk)
            xlist, ylist = np.unravel_index(np.where(er.ravel()==True),er.shape)#atom place

    def deconvolution(self):
        """
        deconvolution for only in 10*10 sites?
        """
        pass

    def latticefind(self):
        """
        lattice site and beam position calibration???
        once in a few times?
        """
        pass

    def saveimg(self,num):
        """
        save image and atom position, and then image after 
        """
        np.savetxt(str(num)+".csv",self.data)
        np.savetxt(str(num)+".csv",np.concatenate(self.xlist,self.ylist))
        return True

    def saveafter(self):
        np.savetxt(str(num)+".csv",self.data2)
        return True

    #############################################
    """ hologram """
    #############################################
    def prepareholograms(self,xsites,ysites):
        """
        calc and save holograms(prob distributions) for lattice sites inside 10*10 area
        maybe, simply read .np file???
        need re-calibration of lattice configulation every few times
        """
        for i in xrange(xsites):
            for j in xrange(ysites):
                self.preholograms[i,j,:,:]=self.hologramize(self.phm,i*self.latticex+self.latticex0,j*self.latticey+self.latticey0,True,np.array([0]),np.array([0]),P=1,alpha=8.5)

    def superpose(self,xlist,ylist,P=1):
        """
        implementation using for loop -> need to be modified for speed
        """
        value=np.zeros((1080,1080))
        dmdpattern = np.zeros((1080,1920),dtype=np.int)
        for i in xrange(xlist.shape[0]):
            value+=self.preholograms[xlist[i],ylist[i]]
        dmdpattern[0:1080,420:1500]=ne.evaluate("1*(value/iteration <P)*(X**2+Y**2<620**2)",local_dict={'iteration':xlist.shape[0],'P':P,'value':value,'X':self.X,'Y':self.Y})
        #return dmdpattern
        self.image2 = pygame.surfarray.make_surface(dmdpattern.T*255)  
        self.swap_event.set() 


    def prehologramize(self,phm,Xshift,Yshift,correction,posX,posY,P=1,alpha=8.5):
        iteration = posX.shape[0]
        dmdpattern = np.zeros((1080,1920),dtype=np.int)
        #rand = np.random.rand(self.Xsize,self.Ysize)
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
        return value

    def hologramize(self,phm,Xshift,Yshift,correction,posX,posY,P=1,alpha=8.5):
        iteration = posX.shape[0]
        dmdpattern = np.zeros((1080,1920),dtype=np.int)
        #rand = np.random.rand(self.Xsize,self.Ysize)
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
        #dmdpattern[0:1080,420:1500]=ne.evaluate("1*(value <P)",local_dict={'value':value,"P":P})
        
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
            fimage[0] = np.abs(self.lgaus(width,0,1))
            fimage[1] = np.angle(self.lgaus(width,0,1))#math.pi*(lgaussian(targetshape,width)<0)
        elif (image == "circles"):
            fimage[0] = 1*(self.X**2+self.Y**2 <100)+((self.X-100)**2+self.Y**2 <100)
        #elif (image == "circles"):
        #    fimage[0] = 1*(self.X**2+self.Y**2 <100)+((self.X-100)**2+self.Y**2 <100)
        elif (image=="confine"):
            #im = gaus(self.X,self.Y,0.5)#1*(self.X**2+self.Y**2<100)*(-(self.X/10)**2-(self.Y/10)**2)# + lgaus(self.X,self.Y,width*0.01,0,1)
            #fftim = np.fft.fftshift(np.fft.fft2(im))
            fimage[0] = np.absolute(-1*self.gaus(0.7)+self.lgaus(width,0,1))
            fimage[1] = np.angle(self.lgaus(self.X,self.Y,width,0,1))
            fimage[1,::2,1::2] = fimage[1,::2,::2]
            fimage[1,1::2,::2] = fimage[1,::2,::2]
            fimage[1,1::2,1::2] = fimage[1,::2,::2]
        else:
            im =np.array(pil.open(image))[:,:,1]/255. 
            fftim = np.fft.fftshift(np.fft.fft2(im))
            fimage[0] = np.absolute(fftim)
            fimage[1] = np.angle(fftim)
            fimage[1,::2,1::2] = fimage[1,::2,::2]
            fimage[1,1::2,::2] = fimage[1,::2,::2]
            fimage[1,1::2,1::2] = fimage[1,::2,::2]
            
        return fimage
    

    def imload(path):
        return np.array(pil.open(path))[:,:,1]/255.

    def gaus(self,width):
        return 1*np.exp(-(self.X**2+self.Y**2)/(2.*(width*2)**2))

    def tem(xx,yy,width):
        order = np.zeros((4,4))
        order[3,3] =1
        width *= xx.shape[0]
        #return 1*np.polynomial.hermite.hermval(np.sqrt(2)*xx/width,order)*np.polynomial.hermite.hermval(np.sqrt(2)*yy/width,order2)*np.exp(-(xx**2+yy**2)/width)
        return 1*np.polynomial.hermite.hermval2d(np.sqrt(2)*xx/width,np.sqrt(2)*yy/width,order)*np.exp(-(xx**2+yy**2)/width**2)

    def lgaus(xx,yy,width):
        order=4
        lorder = 0
        width *= xx.shape[0]*100
        r=np.sqrt(xx**2+yy**2)
        laguerre = special.genlaguerre(order,lorder)
        return 1*(r**lorder)*laguerre(2*r**2/width**2)*np.exp(-r**2/width**2)*np.exp(1j*lorder*math.pi) #replace 1 by phi


    def sinc(xx,yy,width):
        width *= xx.shape[0]/1.7
        return np.sinc(np.sqrt(xx**2+yy**2)/width)

    def linsinc(xx,yy,width):
        width *= xx.shape[0]/1.7
        return np.sinc((xx-yy)/width)



def imread(path):
    f=open(path,'rb')
    header = []
    header.append(f.read(4)) #fileid
    header.append(unpack('H',f.read(2))) #heddersize
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(unpack('H',f.read(2)))
    header.append(f.read(40)) #name
    header.append(f.read(100))
    header.append(unpack('i',f.read(4)))
    header.append(unpack('i',f.read(4)))
    header.append(unpack('h',f.read(2)))
    header.append(unpack('h',f.read(2)))
    header.append(unpack('h',f.read(2)))
    header.append(unpack('h',f.read(2)))
    header.append(unpack('i',f.read(4)))

    data = np.zeros((1,header[5][0]-header[3][0]+1,header[6][0]-header[4][0]+1))
    formatting = ''
    for j in xrange(header[5][0]-header[3][0]+1):
        formatting += 'H'
    for i in xrange(header[6][0]-header[4][0]+1):
        data[0,i,:]=np.array(unpack(formatting,f.read(2*(header[5][0]-header[3][0]+1))))-header[16][0]
    return  header, data

def readdatas(datanum):

    filenum = 5
    header = []

    for i in xrange(datanum):
        for j in xrange(filenum):
            if np.logical_and(i==0,j==0):
                head,data = imread("./../../CCD/160208/data0"+str(i+10)+"/update00"+str(j+1)+".pmi")  
                header.append(head)
            else:
                head,dat = imread("./../../CCD/160208/data0"+str(i+10)+"/update00"+str(j+1)+".pmi")
                header.append(head)
                data = np.append(data,dat,axis=0)
    return data

def readdata(datanum,imgnum,path ="./../../CCD/160208" ):
    head,data = imread(path+"/data0"+str(datanum)+"/update00"+str(imgnum)+".pmi")
    return data[0]

def showfound(data,op):
    image = np.zeros((512,512,3),dtype=np.float32)
    color = 1.*(data-1*np.amin(data))/(np.amax(data)-np.amin(data))
    image[:,:,0] = color
    image[:,:,1] = 1.*np.logical_not(op)*color
    image[:,:,2] = 1.*np.logical_not(op)*color
    plt.imshow(image,interpolation="None")



if __name__ == '__main__':

    a=feedback()
    xlist=np.array([0,1,2,3,4,5,6,7,8,9])
    ylist=np.array([0,1,2,3,4,5,6,7,8,9])
    cProfile.run("a.superpose(xlist,ylist)")