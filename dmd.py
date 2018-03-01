"""
DMD HDMI control program using pygame.
author: Shinichi Sunami
date:12/12/2016
"""

import visa
import time, threading
import pygame, sys
from struct import *
from pygame.locals import *
import numpy as np
import math


class disp():
    def __init__(self):
        # set up pygame
        pygame.init()
#        self.windowSurface = pygame.display.set_mode((1920, 1080), FULLSCREEN)# full HD
        self.windowSurface = pygame.display.set_mode((1000, 800))# for laptop
        self.image = pygame.Surface((1920,1080)) 
        self.image2 = pygame.image.load("./hologram2.jpg").convert() 
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(0)
        self.stop_event = threading.Event() #flag stop or not 
        self.swap_event = threading.Event()  #flag increment or not
        self.count=0
        self.count2=0

        #create pygame thread and start
        self.thread = threading.Thread(target = self.showing)
        self.thread.start()

        #pygame.surfarray.blit_array()
        '''you can blit array directly which is faster. values must be [0,255] and 8 depth'''

    def getinput(self):
        """ wait for z&esc command and stop the program"""
        pygame.event.pump()
        key = pygame.key.get_pressed()
        if key[K_z] & key[K_ESCAPE] :
            self.stop()
            print "stop"

    def stop(self):
        """stop thread"""
        self.stop_event.set()
        pygame.quit()
        self.thread.join() 
        self.stop_trig=True
        print "finished. roop count: " + str(self.count) + " , " + str(self.count2)

    def imgswap(self):
        """ swap image to new one """
        for i in xrange(10):
            image = "linsinc"        
            self.image2 = self.hologramize(0.01,-0.01)
            self.swap_event.set() 
            image = "gaus"        

            self.image2 = self.hologramize(0.01,0.02)
            self.swap_event.set() 

    def darkim(self):
        """ show dark image """
        darkim = np.zeros((1920,1080))
        #pygame.event.get()
        self.image2 = pygame.surfarray.make_surface(darkim*255) 
        self.windowSurface.blit(self.image2,(0,0))#,(50,0,1000,1000))
        pygame.display.flip()

    def showing(self):
        """ run the pygame loop """
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

    def imgswap_atom(self,num,start):
            #plt.ion()
            #plt.figure()
            #plt.show()
            for i in xrange(num):
                self.hologram_atoms(i+start)
                #self.image2 = pygame.surfarray.make_surface(self.hologram_atoms(i).T*255)
                self.swap_event.set() 

