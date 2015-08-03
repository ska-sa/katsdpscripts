#!/usr/bin/python

##Basic socket interface to the R&S signal generator used for CW test signal input

import socket,  time

class SCPI:
  PORT = 5025
  BAUDRATE = 9600

  ## Connect to the R&S signal generator
  def __init__(self,
               host=None, port=PORT,           # set up socket connection
               device=None, baudrate=BAUDRATE, # set up serial port not used
               timeout=1,
               display_info = False):
    if device:
      raise RuntimeError('Only one connection can be initaited at a time.\nSelect socket connection.\n')

    # Ethernet socket connection
    self.connection = None
    if host:
      self.connection = 'socket'
      self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.s.connect((host, port))
      self.s.settimeout(1)
    else:
      raise RuntimeError('No connections specified.\n')

    # Querie instrument identificaton
    if display_info:
        self.write("*IDN?")
        print "DEVICE: " + self.read()

  def display_info(self):
    self.write("*IDN?")
    return  "DEVICE: " + self.read()

  def testConnect(self):
    try:   
      self.write('*IDN?')
      return self.read()
    except:
      return False

  # send query / command via relevant port comm
  def write(self, command):
    self.s.send(command+ '\n')
    time.sleep(1)

  def read(self):
    return self.s.recv(128)

  # activates RF output
  def outputOn(self):
       self.write("OUTPut ON")
  # deactivates the RF output
  def outputOff( self):
       self.write("OUTPut OFF")
  # reset
  def reset(self):
    self.write("*RST")
    self.write(" *CLS")
    time.sleep(5)
    # Sleep for 5 second the time of the reset

  # close the comms port to the R&S signal generator
  def  __close__(self):
    self.s.close()

  # set requested frequency
  def setFrequency(self, freq):
    self.write(" FREQuency %.2f"%(freq,)) # Hz
  
  def setSweep(self, start_freq, step_size, stop_freq, 
        SG_level, dwell_time):
    self.write("SYST:DISP:UPD OFF")
    self.write("FREQ:STAR %.2f kHz"%start_freq)
    self.write("FREQ:STOP %.2f kHz"%stop_freq)
    self.write("SWE:SPAC LIN")
    self.write("SWE:STEP:LIN %f kHz"%step_size)
    self.write("SWE:DWEL %.4f ms"%dwell_time)
    self.write("SWE:MODE AUTO")
    self.write("POW %.1f"%SG_level)
    self.write("FREQ:MODE SWE")


  # read signal generator frequency
  def getFrequency(self):
    self.write('FREQuency?')
    return_freq=self.read()
    try:
      return_freq=float(return_freq)
    except Exception as e:
      print e
      print return_freq.split('\n')
    return return_freq # Hz

  # set requested power level
  def setPower(self, pwr):
    self .write('POWer %s'%str(pwr)) # dBm

  # read sig gen power level
  def getPower(self):
    self.write('POWer?')
    return float(self.read()) # dBm
    


if __name__ == '__main__':

  # SMB100A R&S Signal Generator IP address
  siggen_ip='192.168.14.61'
  siggen_port=5025


## Using SCPI class for comms to signal generator for CW input signal
  sigme=SCPI(siggen_ip)
  sigme.setSweep(100, 1000, 1e6, -25, 100)
  sigme.outputOn()
  try:
    sigme.__close__()
    print 'Closing all ports...'
  except:
    pass # socket already closed
#fin
