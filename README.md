# DMDControl
## Digital Micromirror Device fast-fourier hologram generator
   * optimised speed using cython (feedback.pyx)
   * latest version yet to be pushed
   * presented at http://ico24.org/ abstract is at https://confit.atlas.jp/guide/event/ico24/subject/W1G-02/tables?cryptoId=
   * include Andor EMCCD wrapper
   * non-cython version is optimized using numexpr parallel-processing library

## Requirements
   * ANDOR DLL (for feedback.pyx or andor.py)
   * NI-DAQmx DLL
   * pygame, pyDAQmx, numexpr

## experiment setup
   * TIDL6500 DMD (connected via usb and displayport with PC. board tweaked to suppress 10KHz noise https://arxiv.org/abs/1611.03397)      
   * NI DAQ (USB DAQ is fine) 
   * simple circuit for https://arxiv.org/abs/1611.03397 as in pdf file with PICAXE 28X2 programmed using BASIC

## program usage
programming a sequence such as triggering, loop, hologram generation can be done in `dmdmain` class in main.py. 

### test run
`python main.py`
it displays a gaussian and loops over different defocus coefficients.

### functions used in `main.py`

