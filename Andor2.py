from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import visa
import time
import os
import math
import pickle
from scipy import optimize



DRV_SUCCESS = 20002
DRV_VXDNOTINSTALLED = 20003

dll = windll.LoadLibrary("C:/Program Files/Andor SOLIS/Drivers/atmcd32d.dll")
c_long_p=POINTER(c_long)
c_uint_p=POINTER(c_uint)
c_int_p=POINTER(c_int)
c_float_p=POINTER(c_float)
c_long_arr = c_long*262144
c_long_arrp = POINTER(c_long_arr)


dll.Initialize.argtypes=[c_char_p]
dll.Initialize.restype=c_uint
dll.GetAcquiredData.argtypes=[c_long_p,c_ulong]
dll.GetAcquiredData.restype=c_uint
dll.ShutDown.restype=c_uint
dll.GetNumberADChannels.argtypes=[c_int_p]
dll.GetNumberADChannels.restype=c_uint
dll.GetHSSpeed.restype=c_uint
dll.GetHSSpeed.argtypes=[c_int,c_int,c_int,c_float_p]
dll.GetDetector.restype=c_uint
dll.GetDetector.argtypes=[c_int_p,c_int_p]
#dll.GetNewData.argtypes = [c_long_arrp, c_int]
#dll.GetNewData.restype=c_uint
dll.WaitForAcquisition.restype=c_uint
dll.AbortAcquisition.restype=c_uint
dll.PrepareAcquisition.restype=c_uint
dll.SetShutter.restype=c_uint
dll.SetShutter.argtypes = [c_int, c_int,c_int,c_int]
dll.StartAcquisition.restype=c_uint
dll.SendSoftwareTrigger.restype=c_uint
dll.SetImage.restype=c_uint
dll.SetImage.argtypes = [c_int, c_int,c_int,c_int,c_int,c_int]
dll.GetAcquisitionTimings.argtypes=[c_float_p,c_float_p,c_float_p]
dll.GetAcquisitionTimings.restype=c_uint
dll.SetTriggerMode.restype=c_uint
dll.SetTriggerMode.argtypes=[c_int]

dll.SetExposureTime.argtypes=[c_float]
dll.SetExposureTime.restype=c_uint
dll.GetFastestRecommendedVSSpeed.argtypes=[c_int_p,c_float_p]
dll.GetFastestRecommendedVSSpeed.restype=c_uint
dll.GetNewData.argtypes=[c_long_p,c_long]
dll.GetNewData.restype=c_uint
cdll.msvcrt.malloc.restype=c_long_p

dll.SetAcquisitionMode.restype=c_uint
dll.SetAcquisitionMode.argtypes=[c_int]
dll.IsTriggerModeAvailable.argtypes=[c_int]
dll.IsTriggerModeAvailable.restype=c_uint

dll.SetExposureTime.restype=c_uint
dll.SetExposureTime.argtypes=[c_float]
dll.SetVSSpeed.restype=c_uint
dll.SetVSSpeed.argtypes=[c_int]
dll.GetNumberVSSpeeds.restype=c_uint
dll.GetNumberVSSpeeds.argtypes=[c_int_p]

##
dll.SetEMGainMode.restypes=c_uint
dll.SetEMGainMode.argtypes=[c_int]
dll.SetEMCCDGain.restypes=c_uint
dll.SetEMCCDGain.argtypes=[c_int]
dll.GetEMCCDGain.restypes=c_uint
dll.GetEMCCDGain.argtypes=[c_int_p]

dll.SetTemperature.argtypes=[c_int]
dll.SetTemperature.restype=c_int
dll.GetTemperature.argtypes=[c_int_p]
dll.GetTemperature.restype=c_int
dll.IsCoolerOn.argtypes=[c_int_p]
dll.CoolerON.restype=c_int
dll.CoolerOFF.restype=c_int

dll.GetBaselineClamp.restypes=c_int
dll.GetBaselineClamp.argtypes=[c_int_p]
dll.SetBaselineClamp.restypes=c_uint
dll.SetBaselineClamp.argtypes=[c_int]
dll.SetBaselineOffset.restypes=c_int
dll.SetBaselineOffset.argtypes=[c_int]

dll.AbortAcquisition.restypes=c_int

dll.GetBackground.argtypes=[c_long_p,c_long]
dll.GetBackground.restype=c_uint


class Andor(object):
    def __init__(self, **kwargs):
        path = "."
        c_s             =   c_char_p(path)
        self._iAD       =   c_int(0)
        self._nAD       =   c_int(0)
        self._nAD_p     =   c_int_p(self._nAD)
        self._HSS       =   c_float(0)
        self._HSS_p     =   c_float_p(self._HSS)
        self._giSize    =   c_int(992*992)
        self._extime    =   c_float(0.1)
        self._extime_p  =   c_float_p(self._extime)
        self._acc       =   c_float(0)
        self._acc_p     =   c_float_p(self._acc)
        self._kin       =   c_float(0)
        self._kin_p     =   c_float_p(self._kin)
        self._trgmode   =   c_int(10)   # trgmode10 : Software Trigger
        self._acqmode   =   c_int(1)    # acqmode=1  : Single Shot
        self._VSSpeed   =   c_int(0)
        self._EMGmode   =   c_int(2)
        self._EMGmode_p =   c_int_p(self._EMGmode)
        self._EMGain    =   c_int(10)
        self._EMGain_p  =   c_int_p(self._EMGain)
        self._gblXpixels = c_int(992)
        self._gblYpixels = c_int(992)
        self._image = np.zeros((992,992),dtype=int)
        
        self._camtemperature    = c_int(0)
        self._camtemperature_p  = c_int_p(self._camtemperature)
        self._coolerstate       = c_int(1)
        self._coolerstate_p     = c_int_p(self._coolerstate)
        self._setpoint          = -70

        self._clampstate    = c_int(0)
        self._clampstate_p  = c_int_p(self._clampstate)


        """initialize"""
        err=dll.Initialize(c_s)
        if err==DRV_SUCCESS:
            print "initialization success"
        else:
            print "initialization error %d"%(err)
            

        """PrepareAcquisition"""
        err=dll.PrepareAcquisition()
        if err==DRV_SUCCESS:
            print "PrepareAcquisition success"
        else:
            print "PrepareAcquisition error %d"%(err)
        

        """SetShutter"""
        err=dll.SetShutter(c_int(1),c_int(1),c_int(0),c_int(0))
        if err==DRV_SUCCESS:
            print "SetShutter success"
        else:
            print "SetShutter ERROR %d"%(err)
            

        """getADnumber"""
        err=dll.GetNumberADChannels(self._nAD_p)
        print "nAD %d"%(self._nAD.value)
        

        """getHSSpeed"""
        err=dll.GetHSSpeed(self._iAD,c_int(0),c_int(1),self._HSS_p)
        print "HSSpeed %f"%(self._HSS.value)
        

        """SetAcquisitionMode"""
        err=dll.SetAcquisitionMode(self._acqmode)
        if err==DRV_SUCCESS:
            print "Set to acquire Single Shot mode"
        else:
            print "SetAcquisitionMode error %d"%(err)

        """SetTriggerMode"""
        err=dll.SetTriggerMode(self._trgmode)
        if err==DRV_SUCCESS:
            print "set to Sofware trigger mode"
        else:
            print "triggermode ERROR %d"%(err)

        """SetImage"""
        err=dll.SetImage(c_int(1),c_int(1),c_int(1),self._gblXpixels,c_int(1),self._gblYpixels);
        if err==DRV_SUCCESS:
            print "SetImage success"
        else:
            print "SetImage ERROR %d"%(err)

        """Set Cooler"""
        err=dll.CoolerON()
        err=dll.SetTemperature(self._setpoint)
        dll.GetTemperature(self._camtemperature_p)
        if err==DRV_SUCCESS:
            print "Set Temperature success"
        else:
            print "SetTemperature ERROR %d"%(err)

        """SetExposureTime"""
        err=dll.SetExposureTime(self._extime)
        if err==DRV_SUCCESS:
            print "SetExposureTime success"
        else:
            print "SetExposure ERROR %d"%(err)

        """SetVSSpeed"""
        dll.GetNumberVSSpeeds(c_int_p(self._VSSpeed))
        err=dll.SetVSSpeed(c_int(self._VSSpeed.value-2))
        if err==DRV_SUCCESS:
            print "setVS success %d"%(self._VSSpeed.value)
        else:
            print "setVS ERROR %d"%(err)

        """SetEMGainMode"""
        err=dll.SetEMGainMode(self._EMGmode)
        if err==DRV_SUCCESS:
            print "SetEMGainMode liner"
        else:
            print "SetEMGainMode ERROR %d"%(err)

        """SetEMCCDGain"""
        err=dll.SetEMCCDGain(self._EMGain)
        if err==DRV_SUCCESS:
            print "SetEMCCDGain %d"%(self._EMGain.value)
        else:
            print "SetEMCCDGain ERROR %d"%(err)

        """GetAcquisitionTimings"""
        dll.GetAcquisitionTimings(self._extime_p,self._acc_p,self._kin_p);
        print "Exposure time %f s"%(self._extime.value)
        print "Accumulate %f"%(self._acc.value)
        print "Kinetic %f"%(self._kin.value)
        
        """SetBaselineClamp"""
        err=dll.SetBaselineClamp(self._clampstate)
        if err==DRV_SUCCESS:
            print "SetBaselineClamp success"
        else:
            print "SetBaselineClamp ERROR %d"%(err)

        #"""StartAcquisition"""
        #err=dll.StartAcquisition()
        #if err==DRV_SUCCESS:
        #    print "startAcquisition success"
        #else:
        #    print "startAcquisition error"

    def shutdown(self):
        """AbortAcquisition"""
        err = dll.AbortAcquisition()
        """shutdown"""
        err = dll.ShutDown()
        if err==DRV_SUCCESS:
        	print "shutdown success"
        else:
        	print "shutdown error"

    @property
    def exposuretime(self):
        return self._extime.value
    @exposuretime.setter
    def exposuretime(self,value):
        err=dll.SetExposureTime(c_float(value))
        if err==DRV_SUCCESS:
            self._extime=c_float(value)
        else:
            print "SetExposure ERROR %d"%(err)

    @property
    def setpoint(self):
        return self._setpoint
    @setpoint.setter
    def setpoint(self, value):
        err=dll.SetTemperature(c_int(value))
        if err==DRV_SUCCESS:
            self._setpoint=value
        else:
            print "Set setpoint ERROR %d"%(err)

    @property
    def cooler(self):
        dll.IsCoolerOn(self._coolerstate_p)
        return self._coolerstate.value
    @cooler.setter
    def cooler(self,state):
        if state:
            dll.CoolerON()
        else:
            dll.CoolerOFF()

    def readtemp(self):
        dll.AbortAcquisition()
        err=dll.GetTemperature(self._camtemperature_p)
        print err
        return {"temperature":self._camtemperature.value}

    @property
    def emgain(self):
        err=dll.GetEMCCDGain(self._EMGain_p)
        print err
        return self._EMGain.value
    @emgain.setter
    def emgain(self,value):
        err=dll.SetEMCCDGain(c_int(value))
        if err==DRV_SUCCESS:
            self._EMGain=c_int(value)
        else:
            print err

    @property
    def extime(self):
        err=dll.GetAcquisitionTimings(self._extime_p,self._acc_p,self._kin_p)
        #print err
        return self._extime.value
    @extime.setter
    def extime(self,value):
        err=dll.SetExposureTime(c_float(value))
        if err==DRV_SUCCESS:
            self._extime=c_float(value)
        else:
            print err
                                      
                
    def getimage(self):
        err=dll.StartAcquisition()
        if err==DRV_SUCCESS:
            dll.SendSoftwareTrigger()
            dll.WaitForAcquisition()
            dll.GetNewData(self._image.ctypes.data_as(c_long_p),self._giSize)
        else:
            print "startAcquisition error"

    def showimages(self,klen):
        for i in xrange(klen):
            self.getimage()
            plt.figure()
            plt.gray()
            plt.imshow(self._image)
            plt.colorbar()
        #plt.ion()
        plt.show()

    def settings(self):
        dll.AbortAcquisition()
        err=dll.GetTemperature(self._camtemperature_p)
        return {"iAD":self._iAD.value,"nAD":self._nAD.value,"HSS":self._HSS.value,
                "VSS":self._VSSpeed.value,"ExTime":self.extime,
                "TRG":self._trgmode.value,"ACQmode":self._acqmode.value,
                "EMGmode":self._EMGmode.value,"EMgain":self._EMGain.value,
                "temp":self._camtemperature.value}
        

        
class OscilloScope:
        def __init__(self):
                rm = visa.ResourceManager()
                self.osc = rm.open_resource("USB0::0x0699::0x03A2::C011547::INSTR")
                print self.osc.ask("*IDN?")
                
                self.span=[0,5,0.1,5]
                self.position=[0,0,-3,0]
                self.timelimit=10000

                self.osc.write("CH1:VOL %f"%(self.span[1]))
                print "set CH1 span %f in init"%(self.span[1])

                self.osc.write("CH2:VOL %f"%(self.span[2]))
                print "set CH2 span %f in init"%(self.span[2])

                self.osc.write("CH3:VOL %f"%(self.span[3]))
                print "set CH3 span %f in init"%(self.span[3])


                self.osc.write("CH1:POS %f"%(self.position[1]))
                print "set CH1 position %f in init"%(self.position[1])
                self.osc.write("CH2:POS %f"%(self.position[2]))
                print "set CH2 position %f in init"%(self.position[2])
                self.osc.write("CH3:POS %f"%(self.position[3]))
                print "set CH2 position %f in init"%(self.position[3])
                
                print "set timelimit %f in init"%(self.timelimit)

                self.osc.write("DAT:WID 1")
                self.timerangestart=1200
                self.osc.write("DAT:STAR %d"%(self.timerangestart))
                self.timerangestop=2500
                self.osc.write("DAT:STOP %d"%(self.timerangestop))
                self.osc.write("CH1:PRO 1")
                self.osc.write("CH2:PRO 1")
                self.osc.write("CH3:PRO 1")
                self.osc.write("TRIG:MAI:EDGE:SOU CH1")
                self.osc.write("TRIG:MAI:MOD NORM")
                self.osc.write("TRIG:MAI:LEV 1")
                self.hscale=0.05
                self.osc.write("HOR:MAI:SCA %f"%(self.hscale))

                
                self.osc.timeout=20000

   
        def getspan(self,ch):
                return self.span[ch]

        def setspan(self,ch,span):
                self.span[ch]=span
                self.osc.write("CH%d:VOL %f"%(ch,self.span[ch]))
                print "set CH%d span %f in func"%(ch,self.span[ch])

        def getposition(self,ch):
                return self.position[ch]

        def setposition(self,ch,pos):
                self.position[ch]=pos
                print "set CH%d position %f in func"%(ch,self.position[ch])
                
        def settimelimit(self,lim):
                self.timelimit=lim
                print "ser timelimit %f in func"%(self.timelimit)

        def takedata(self,ch):
                self.osc.write("DAT:SOU CH%d"%(ch))
                dstr=self.osc.ask("CURV?")
                darr=np.array([int(i) for i in dstr.split(",")])
                vol=(darr/25.0-self.position[ch])*self.span[ch]
                return vol


class FunctionGenerator():
        
        def __init__(self):
        
                rm = visa.ResourceManager()
                self.fg = rm.open_resource("USB0::0x1AB1::0x0641::DG4D141700511::INSTR")
                self._offs=[0,2.0,3.0]
                self.fg.write(":SOUR1:FUNC DC;SOUR1:VOLT:OFFS %fV"%(self._offs[1]))
                self.fg.write(":SOUR2:FUNC DC;SOUR2:VOLT:OFFS %fV"%(self._offs[2]))
        def on(self,ch):
                self.fg.write("OUTP%d ON"%(ch)) #ch1:control,ch2:TTL
        def off(self,ch):
                self.fg.write("OUTP%d OFF"%(ch))
        def volt(self,ch,volt):
                self._offs[ch]=volt
                self.fg.write("SOUR%d:VOLT:OFFS %fV"%(ch,self._offs[ch]))

    

def savedatasig(ancl,oscl,fgcl,path,number):
    fgcl.on(2)
    ancl.getimage()
    np.save(path+"imagesig%05d.npy"%(number),ancl._image)
    state=ancl.settings()
    state.update({"time":time.time(),"osTstart":oscl.timerangestart,
                  "osTstop":oscl.timerangestop,"osTscale":oscl.hscale,
                  "ctrlV":fgcl._offs[1]})
    f=open(path+"statesig%05d.txt"%(number),"w")
    pickle.dump(state,f)
    f.close()
    time.sleep(0.3)
    fire=oscl.takedata(1)
    pd=oscl.takedata(2)
    ttl=oscl.takedata(3)
    np.save(path+"firesig%05d"%(number),fire)
    np.save(path+"pdsig%05d"%(number),pd)
    np.save(path+"ttlsig%05d"%(number),ttl)

def savedatadark(ancl,oscl,fgcl,path,number):
    fgcl.off(2)
    ancl.getimage()
    np.save(path+"imagedark%05d.npy"%(number),ancl._image)
    state=ancl.settings()
    state.update({"time":time.time(),"osTstart":oscl.timerangestart,
                  "osTstop":oscl.timerangestop,"osTscale":oscl.hscale,
                  "ctrlV":fgcl._offs[1]})
    f=open(path+"statedark%05d.txt"%(number),"w")
    pickle.dump(state,f)
    f.close()
    time.sleep(0.3)
    fire=oscl.takedata(1)
    pd=oscl.takedata(2)
    ttl=oscl.takedata(3)
    np.save(path+"firedark%05d"%(number),fire)
    np.save(path+"pddark%05d"%(number),pd)
    np.save(path+"ttldark%05d"%(number),ttl)

    
def saveloop(ancl,oscl,fgcl,path,N):
    tr=int(N*8.33)
    print "require about %d hours %d minutes"%(tr/3600,(tr%3600)/60)
    fgcl.on(2)
    fgcl.off(2)
    for i in range(N):
        savedatasig(ancl,oscl,fgcl,path,i)
        savedatadark(ancl,oscl,fgcl,path,i)
        if i%500==499:
            fgcl.off(1)
            time.sleep(1)
            fgcl.on(1)
            


        
def savedatafilter(ancl,oscl,path,number,crit,accu):
    ancl.getimage()
    #np.save(path+"image%05d.npy"%(number),ancl._image)
    state=ancl.settings()
    state.update({"time":time.time(),"osTstart":oscl.timerangestart,
                  "osTstop":oscl.timerangestop,"osTscale":oscl.hscale})
    f=open(path+"state%05d.txt"%(number),"w")
    pickle.dump(state,f)
    f.close()
    time.sleep(0.5)
    fire=oscl.takedata(1)
    pd=oscl.takedata(2)
    np.save(path+"fire%05d"%(number),fire)
    np.save(path+"pd%05d"%(number),pd)
    ave=pd[np.where(fire>4)][1:-1].mean()
    if (ave>crit*(1-accu))and(ave<crit*(1+accu)):
        np.save(path+"image%05d.npy"%(number),ancl._image)
        print "image%05d saved"%(number)

def saveloopfilter(ancl,oscl,path,N,crit,accu):
    for i in range(N):
        savedatafilter(ancl,oscl,path,i,crit,accu)
        time.sleep(0.5)

def pccalib(ancl,oscl,fgcl,N):
    CCD=np.zeros(N)
    ctrl=np.linspace(0,0.6,N)
    ancl.getimage()
    oscl.takedata(1)
    oscl.takedata(2)
    oscl.takedata(3)
    for i in range(N):
        fgcl.on(2)
        fgcl.volt(1,ctrl[i])
        ancl.getimage()
        oscl.takedata(1)
        oscl.takedata(2)
        oscl.takedata(3)
        CCD[i]=ancl._image.sum()
    def func(x,a,b):
        return a*(x-b)
    pa,co=optimize.curve_fit(func,CCD[1:],ctrl[1:])
    xfit=np.linspace(0,CCD[-1],100)
    yfit=func(xfit,pa[0],pa[1])
    plt.plot(CCD,ctrl,"o")
    plt.plot(xfit,yfit,"-",label="ctrl=%e(CCDcount-%e)"%(pa[0],pa[1]))
    plt.xlabel("total CCD count (a.u.)")
    plt.ylabel("control voltage (V)")
    plt.ylim(0,)
    plt.legend(loc="best")
    return CCD,ctrl,pa
        
a=Andor()
os=OscilloScope()
fg=FunctionGenerator()
