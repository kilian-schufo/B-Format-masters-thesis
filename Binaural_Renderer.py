
# coding: utf-8

# In[8]:


import serial
ser = serial.Serial('/dev/tty.usbserial-AH03F9XC')  # open serial port
ser.baudrate = 57600 #if other rate it changes to byte mode


# In[26]:


#load modules
import numpy as np
from __future__ import division
from __future__ import print_function
import argparse
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x
import sys
import threading

#choose any B-format sound file in FuMa format (Channel order W, X, Y, Z)
filename = 'filename.wav'
clientname = 'file_player'
buffersize = 20
manual = False

if buffersize < 1:
    parser.error('buffersize must be at least 1')

q = queue.Queue(maxsize=buffersize)
event = threading.Event()


def print_error(*args):
    print(*args, file=sys.stderr)


def xrun(delay):
    print_error("An xrun occured, increase JACK's period size?")


def shutdown(status, reason):
    print_error('JACK shutdown!')
    print_error('status:', status)
    print_error('reason:', reason)
    event.set()


def stop_callback(msg=''):
    if msg:
        print_error(msg)
    for port in client.outports:
        port.get_array().fill(0)
    event.set()
    raise jack.CallbackExit

   
    

def process(frames):
    if frames != blocksize:
        stop_callback('blocksize must not be changed, I quit!')
    try:
        data = q.get_nowait()
    except queue.Empty:
        stop_callback('Buffer is empty: increase buffersize?')
    if data is None:
        stop_callback()  # Playback is finished
    
    lines = str(ser.read(ser.in_waiting)).split('#') #list with all returned lines from head tracker
    if ',' in lines[-1]:
        yaw = float(lines[-1].split('=')[1].split(',',1)[0])
    else:
        yaw = float(lines[-2].split('=')[1].split(',',1)[0])
    
    if my_buffer.count == 0:
        my_buffer.offset = yaw
    my_buffer.count += 1
    
    angle = -(yaw-my_buffer.offset)
    rad = angle*np.pi/180
    XX = np.sin(rad)*data.T[2,:]+np.cos(rad)*data.T[1,:]
    YY = np.cos(rad)*data.T[2,:]-np.sin(rad)*data.T[1,:]
    
    #print('Angle:',angle)
    
    #spherical speaker array
    D45 = driver_signal(2,70*np.pi/180,np.pi/4,data.T[0,:],YY,data.T[3,:],XX)*20000
    D45_bl_L = np.convolve(D45,L45)
    D45_bl_R = np.convolve(D45,R45)
    D135 = driver_signal(2,115*np.pi/180,3*np.pi/4,data.T[0,:],YY,data.T[3,:],XX)*20000
    D135_bl_L = np.convolve(D135,L135)
    D135_bl_R = np.convolve(D135,R135)
    D225 = driver_signal(2,65*np.pi/180,5*np.pi/4,data.T[0,:],YY,data.T[3,:],XX)*20000
    D225_bl_L = np.convolve(D225,L225)
    D225_bl_R = np.convolve(D225,R225)
    D315 = driver_signal(2,110*np.pi/180,7*np.pi/4,data.T[0,:],YY,data.T[3,:],XX)*20000
    D315_bl_L = np.convolve(D315,L315)
    D315_bl_R = np.convolve(D315,R315)
    
    
    client.outports[0].get_array()[:] = np.hstack((D45_bl_L[0:sizeHRIR-1]+my_buffer.BL45,D45_bl_L[sizeHRIR-1:blocksize]))+np.hstack((D315_bl_L[0:sizeHRIR-1]+my_buffer.BL315,D315_bl_L[sizeHRIR-1:blocksize]))+np.hstack((D135_bl_L[0:sizeHRIR-1]+my_buffer.BL135,D135_bl_L[sizeHRIR-1:blocksize]))+np.hstack((D225_bl_L[0:sizeHRIR-1]+my_buffer.BL225,D225_bl_L[sizeHRIR-1:blocksize])) #assign first channel to out_1
    client.outports[1].get_array()[:] = np.hstack((D45_bl_R[0:sizeHRIR-1]+my_buffer.BR45,D45_bl_R[sizeHRIR-1:blocksize]))+np.hstack((D315_bl_R[0:sizeHRIR-1]+my_buffer.BR315,D315_bl_R[sizeHRIR-1:blocksize]))+np.hstack((D135_bl_R[0:sizeHRIR-1]+my_buffer.BR135,D135_bl_R[sizeHRIR-1:blocksize]))+np.hstack((D225_bl_R[0:sizeHRIR-1]+my_buffer.BR225,D225_bl_R[sizeHRIR-1:blocksize])) #assign second channel to out_2
    my_buffer.BL45 = D45_bl_L[blocksize:]
    my_buffer.BR45 = D45_bl_R[blocksize:]
    my_buffer.BL135 = D135_bl_L[blocksize:]
    my_buffer.BR135 = D135_bl_R[blocksize:]
    my_buffer.BL225 = D225_bl_L[blocksize:]
    my_buffer.BR225 = D225_bl_R[blocksize:]
    my_buffer.BL315 = D315_bl_L[blocksize:]
    my_buffer.BR315 = D315_bl_R[blocksize:]
   
   
    
#determine driver signals according to equation (2.15) in the thesis
#theta: colatitude, phi: azimuth, R array radius
def driver_signal(R,theta,phi,A00,A1min1,A10,A11):
    nm00 = (1/(2*np.pi*R**2))*np.sqrt(1/(4*np.pi))*(1/(4*np.pi))*A00
    nm1min1 = (1/(2*np.pi*R**2))*np.sqrt(3/(4*np.pi))*(np.sin(theta)*np.sin(phi)/(4*np.pi))*A1min1
    nm10 = (1/(2*np.pi*R**2))*np.sqrt(3/(4*np.pi))*(np.cos(theta)/(4*np.pi))*A10
    nm11 = (1/(2*np.pi*R**2))*np.sqrt(3/(4*np.pi))*(np.sin(theta)*np.cos(phi)/(4*np.pi))*A11
    return nm00+nm1min1+nm10+nm11

#Import HRTF data from Matlab file
import scipy.io as sio
mat_content = sio.loadmat('HRTF_nh163.mat')


L45 = mat_content['aaL441_eq'].flatten() #left ear HRIR for loudspeaker at 45 degree azimuth
R45 = mat_content['aaR441_eq'].flatten() #right ear ...
L135 = mat_content['bbL441_eq'].flatten()
R135 = mat_content['bbR441_eq'].flatten()
L225 = mat_content['ccL441_eq'].flatten()
R225 = mat_content['ccR441_eq'].flatten()
L315 = mat_content['ddL441_eq'].flatten()
R315 = mat_content['ddR441_eq'].flatten()
   
sizeHRIR = len(L45)

class buff:
    pass
    
#instance of class buff
my_buffer = buff()

my_buffer.count = 0
my_buffer.offset = 0
my_buffer.BL45 = np.zeros(len(L45)-1)
my_buffer.BR45 = np.zeros(len(L45)-1)
my_buffer.BL135 = np.zeros(len(L45)-1)
my_buffer.BR135 = np.zeros(len(L45)-1)
my_buffer.BL225 = np.zeros(len(L45)-1)
my_buffer.BR225 = np.zeros(len(L45)-1)
my_buffer.BL315 = np.zeros(len(L45)-1)
my_buffer.BR315 = np.zeros(len(L45)-1)

#clean buffer (fixed to 1020 bytes length) and queue with old data occuring between two calls of this script
while ser.in_waiting == 1020:
    clean_buffer =  ser.read(ser.in_waiting)
    #print('cleaning')   
    
    
try:
    import jack
    import soundfile as sf
    
    client = jack.Client(clientname)

    
    blocksize = client.blocksize
    samplerate = client.samplerate
    client.set_xrun_callback(xrun)
    client.set_shutdown_callback(shutdown)
    client.set_process_callback(process)
  
    with sf.SoundFile(filename) as f:
        for ch in range(2):
            client.outports.register('out_{0}'.format(ch + 1))
        block_generator = f.blocks(blocksize=blocksize, dtype='float32',
                                   always_2d=True, fill_value=0)
        for _, data in zip(range(buffersize), block_generator):
            q.put_nowait(data)  # Pre-fill queue
        with client:
            if not manual:
                target_ports = client.get_ports(is_physical=True, is_input=True, is_audio=True)
                if len(client.outports) == 1 and len(target_ports) > 1:
                    # Connect mono file to stereo output
                    client.outports[0].connect(target_ports[0])
                    client.outports[0].connect(target_ports[1])
                else:
                    for source, target in zip(client.outports, target_ports):
                        source.connect(target)
            timeout = blocksize * buffersize / samplerate
            for data in block_generator:
                q.put(data, timeout=timeout)
            q.put(None, timeout=timeout)  # Signal end of file
            event.wait()  # Wait until playback is finished
except KeyboardInterrupt:
    print('\nInterrupted by user')
except (queue.Full):
    # A timeout occured, i.e. there was an error in the callback
    parser.exit(1)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

