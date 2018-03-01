"""
instrument control, Digital, analog I/O
12/12/2016 author: sunami
"""


import visa
import time, threading
import numpy as np
import math
import pygame
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *    
from PyDAQmx import Task

class oscillo():
    def __init__(self):
        #setup oscillo
        self.rm=visa.ResourceManager()
        self.oscillo = self.rm.list_resources()[0]
        self.osc = self.rm.get_instrument(self.oscillo)
        print(self.osc.ask("*IDN?"))
        self.osc.write("CH1:SCALE 1E0")
        self.osc.write("HORIZONTAL:MAIN:SCALE 2E-3")
        # make the measurement
        print self.readoscillo()

    def readoscillo(self):
        time.sleep(1.4)
        a= self.osc.query_ascii_values('measurement:meas1:value?')[0]
        #while (a == 9.9e+37):
        #    a= self.osc.query_ascii_values('measurement:immed:value?')[0]
        return a


    def getdarko(self):
        darkim = np.zeros((1920,1080))
        #pygame.event.get()
        self.image2 = pygame.surfarray.make_surface(darkim*255) 
        
        #self.swap_event.set() 
        self.windowSurface.blit(self.image2,(0,0))#,(50,0,1000,1000))
        pygame.display.flip()

        time.sleep(.5)
        dark = self.readoscillo()
        print "darkvalue=" + str(dark)    
        return dark




class Daq():
    """argument  is like "Dev1/ai0" """

    """Class to create a multi-channel analog input
    
    Usage: AI = MultiChannelInput(physicalChannel)
        physicalChannel: a string or a list of strings
    optional parameter: limit: tuple or list of tuples, the AI limit values
                        reset: Boolean
    Methods:
        read(name), return the value of the input name
        readAll(), return a dictionary name:value
    """
    def __init__(self,physicalChannel, limit = None, reset = False):
        if type(physicalChannel) == type(""):
            self.physicalChannel = [physicalChannel]
        else:
            self.physicalChannel  =physicalChannel
        self.numberOfChannel = physicalChannel.__len__()
        if limit is None:
            self.limit = dict([(name, (-10.0,10.0)) for name in self.physicalChannel])
        elif type(limit) == tuple:
            self.limit = dict([(name, limit) for name in self.physicalChannel])
        else:
            self.limit = dict([(name, limit[i]) for  i,name in enumerate(self.physicalChannel)])           
        if reset:
            DAQmxResetDevice(physicalChannel[0].split('/')[0] )
        self.taskHandle=[]
        self.configure()
    def configure(self,trigger=0):
        # Create one task handle per Channel
        taskHandles = dict([(name,TaskHandle(0)) for name in self.physicalChannel])
        for name in self.physicalChannel:
            DAQmxCreateTask("",byref(taskHandles[name]))
            DAQmxCreateAIVoltageChan(taskHandles[name],name,"",DAQmx_Val_RSE,
                                     self.limit[name][0],self.limit[name][1],
                                     DAQmx_Val_Volts,None)
        self.taskHandles = taskHandles
        if trigger:
            DAQmxCfgDigEdgeStartTrig (TaskHandles[0], "Dev1/di0/StartTrigger" ,DAQmx_Val_Rising);
    def readAll(self):
        return dict([(name,self.read(name)) for name in self.physicalChannel])

    def getdark(self):
        darkim = np.zeros((1920,1080))
        pygame.event.get()
        self.image2 = pygame.surfarray.make_surface(darkim*255) 
        
        self.swap_event.set() 
        self.windowSurface.blit(self.image2,(0,0))#,(50,0,1000,1000))
        pygame.display.flip()

        time.sleep(0.1)
        dark = self.read()
        print "darkvalue=" + str(dark)    
        return dark

    def readconfig(self,name=None):
        if name is None:
            name = self.physicalChannel[0]
        self.taskHandle = self.taskHandles[name]  
        DAQmxCfgSampClkTiming (self.taskHandle, 0, 1, DAQmx_Val_Rising, DAQmx_Val_ContSamps ,10)                 
        DAQmxCfgDigEdgeStartTrig (self.taskHandle, "PFI1" ,DAQmx_Val_Rising);
        DAQmxStartTask(self.taskHandle)
        self.data = np.zeros((1,), dtype=np.float64)


    def readfin(self):
    	DAQmxStopTask(self.taskHandle)

    def read(self,name = None):
    	time.sleep(0.45)
        read = int32()
        DAQmxReadAnalogF64(self.taskHandle,1,30.0,DAQmx_Val_GroupByChannel,self.data,1,byref(read),None)
        #DAQmxStopTask(self.taskHandle)
        return self.data[0]

    def waittrig(self,timeout=10):
        self.readconfig()
        self.read()        
        self.readfin()

    def digitalon(self):
    	data = np.array([1,0,0], dtype=np.uint8)
        task = Task()
        task.CreateDOChan("/Dev1/port0/line2","",DAQmx_Val_ChanPerLine)
        task.StartTask()
        task.WriteDigitalLines(1,1,DAQmx_Val_WaitInfinitely,DAQmx_Val_GroupByChannel,data,None,None)
        task.StopTask()

    def trig1(self):
        data = np.array([1,0,0], dtype=np.uint8)
        task = Task()
        task.CreateDOChan("/Dev1/port0/line0","",DAQmx_Val_ChanPerLine)
        task.StartTask()
        task.WriteDigitalLines(1,1,1,DAQmx_Val_GroupByChannel,data,None,None)
        task.StopTask()

    def sendtrig(self):
        self.digitalon()
        self.digitaloff()


    def digitaloff(self):
    	data = np.array([0,0,0], dtype=np.uint8)
        task = Task()
        task.CreateDOChan("/Dev1/port0/line0","",DAQmx_Val_ChanPerLine)
        task.StartTask()
        task.WriteDigitalLines(1,1,DAQmx_Val_WaitInfinitely,DAQmx_Val_GroupByChannel,data,None,None)
        task.StopTask()



class agilis():
    def __init__(self):
        #setup agilis
        self.rm=visa.ResourceManager()
        self.piezo = self.rm.list_resources()[3]
        self.osc = self.rm.get_instrument(self.oscillo)
        print(self.osc.ask("*IDN?"))
        self.osc.write("CH1:SCALE 1E0")
        self.osc.write("HORIZONTAL:MAIN:SCALE 2E-3")
        # make the measurement
        print self.readoscillo()



if __name__ == '__main__':
    a=Daq("Dev1/ai0")
    #a.readconfig()
    for i in xrange(4):
        a.waittrig()
        print("trig")


    """
    for i in xrange(3):
        a.readconfig()
        a.read()
        a.digitalon()
        time.sleep(5)
        a.readfin()
        a.digitaloff()
    """
