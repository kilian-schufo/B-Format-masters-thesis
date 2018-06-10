
# coding: utf-8

# In[ ]:


#load modules
import cmath as ma
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import soundfile as sf


# In[ ]:


#returns the distance of two points given in spherical coordinates
def distance(r1,teta1,phi1,r2,teta2,phi2): 
    
    #transformation to carthesian coordinates
    x1 = r1*ma.cos(phi1)*ma.sin(teta1)
    y1 = r1*ma.sin(phi1)*ma.sin(teta1)
    z1 = r1*ma.cos(teta1)
    x2 = r2*ma.cos(phi2)*ma.sin(teta2)
    y2 = r2*ma.sin(phi2)*ma.sin(teta2)
    z2 = r2*ma.cos(teta2)
    #distance calculation
    d = ma.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return d


# In[ ]:


#function to calculate the angle in rad between two vectors in spherical coordinates (radius,colatitude,azimuth)
#gives always the smallest angle between the vectors. It is okay here because the cosine is symmetrical to pi
#and the cardioid polar pattern is also symmetrical
def angle(r1,teta1,phi1,r2,teta2,phi2):
    
    r1=r2=1 #to still give an angle if one vector has zero length
    
    #converts the vectors to carthesian coordinates
    v1 = [r1*ma.cos(phi1)*ma.sin(teta1),r1*ma.sin(phi1)*ma.sin(teta1),r1*ma.cos(teta1)]
    v2 = [r2*ma.cos(phi2)*ma.sin(teta2),r2*ma.sin(phi2)*ma.sin(teta2),r2*ma.cos(teta2)]
    #calculates the angle
    cos_of_angle = np.dot(v1,v2)/(la.norm(v1)*la.norm(v2))
    angle = ma.acos(cos_of_angle)
    return(angle)


# In[ ]:


#returns the B-format signal of the mic for a frequency vector and a given source position

def bformat(a,freq = [],s_pos = []):
     
    #memory allocation output signals
    s_LF = np.zeros((np.size(freq)),dtype=np.complex_)
    s_LB = np.zeros((np.size(freq)),dtype=np.complex_)
    s_RB = np.zeros((np.size(freq)),dtype=np.complex_)
    s_RF = np.zeros((np.size(freq)),dtype=np.complex_)

    # capsule positions 
    R = 0.0147 # radius of the tetrahedron in meters (according to Gerzon)
    tilt = ma.atan(1/ma.sqrt(2))  # tilt of the capsules in rad (according to Farrar)
    
    # spherical coordinates (r,teta,phi)=(radius,colatitude,azimuth)
    # in accordance with Faller and Farrar
    LFU = ([R, ma.pi/2 - tilt, ma.pi/4])  #left front up
    LBD = ([R, ma.pi/2 + tilt, 3*ma.pi/4]) #left back down
    RBU = ([R, ma.pi/2 - tilt, 5*ma.pi/4]) #right back up
    RFD = ([R, ma.pi/2 + tilt, 7*ma.pi/4]) #right front down

    #distance of capsules to source
    d_LF = distance(*LFU,*s_pos)
    d_LB = distance(*LBD,*s_pos)
    d_RB = distance(*RBU,*s_pos)
    d_RF = distance(*RFD,*s_pos)

    #angle between capsule and source
    angLF = angle(*LFU,*s_pos)
    angLB = angle(*LBD,*s_pos)
    angRB = angle(*RBU,*s_pos)
    angRF = angle(*RFD,*s_pos)
        
    for ii in range(np.size(freq)):
            
            f = freq[ii]
            w = 2*ma.pi*f
            c = 340
            k = w/c

            #pressure at the capsules due to the source at s_pos
            p_LF = ma.exp(-1j*k*d_LF)
            p_LB = ma.exp(-1j*k*d_LB)
            p_RB = ma.exp(-1j*k*d_RB)
            p_RF = ma.exp(-1j*k*d_RF)
    
            #signals of the capsules
            s_LF[ii] = (a + a*ma.cos(angLF))*p_LF
            s_LB[ii] = (a + a*ma.cos(angLB))*p_LB
            s_RB[ii] = (a + a*ma.cos(angRB))*p_RB
            s_RF[ii] = (a + a*ma.cos(angRF))*p_RF
            
    #B-format signals
    W = s_LF + s_RB + s_RF + s_LB #omni
    X = s_LF - s_RB + s_RF - s_LB #f.o.e. forward (to positive x)
    Y = s_LF - s_RB - s_RF + s_LB #f.o.e. leftward (to positive y)
    Z = s_LF - s_LB + s_RB - s_RF #f.o.e. upward (to positive z)
    
    return W,X,Y,Z


# In[ ]:


#non-coincidence correction according to Gerzon as introduced by Batke, W attenuated by 3dB
def corr_batke(a, order, B_format_matrix = [],freq = []):

    B_corrected = np.zeros((np.size(freq)),dtype=np.complex_)
    
    for i in range(len(freq)):
        f = freq[i]
        R = 0.0147
        w = 2*ma.pi*f
        c = 340
        k = w/c
        xx = k*R
        
        if f == 0:
            j_0 = 1
            j_0_diff = 0
            j_1 = 0
            j_1_diff = 1/3
        if f > 0:
            j_0 = ma.sin(xx)/(xx)
            j_0_diff = (xx*ma.cos(xx) - ma.sin(xx))/((xx)**2)
            j_1 = ma.sin(xx)/((xx)**2) - ma.cos(xx)/(xx)
            j_1_diff = ((xx**2 - 2)*ma.sin(xx) + 2*xx*ma.cos(xx))/(xx**3)

        V_0 = 1/(a*j_0 - 1j*(1-a)*j_0_diff)
        V_1 = 1/(a*j_1 - 1j*(1-a)*j_1_diff)
    
        if order == 0:
            B_corrected[i] = B_format_matrix[i]*V_0*(1/ma.sqrt(4*ma.pi))*(1/(ma.sqrt(2)))
        if order == 1:
            B_corrected[i] = B_format_matrix[i]*V_1*(1/ma.sqrt(4*ma.pi))*(-1j)
            
    return B_corrected


# In[ ]:


def doubside(WB1):
    #WB_doub is double sided spectrum, all values exept at n=0 and n=N/2+1 are scaled with 1/sqrt(2) 
    #because the F-transform splits the energy in the two sides of the spectrum
    WB1_rev = WB1[::-1][1:len(WB1)-1]/ma.sqrt(2) #reversed, without first and last value, scaled
    WB1_sc = np.hstack((WB1[0],WB1[1:len(WB1)-1]/ma.sqrt(2),WB1[len(WB1)-1])) #every value but the first and last are scaled
    WB1_doub = np.hstack((WB1_sc,np.conjugate(WB1_rev))) #double sided spectrum
    return WB1_doub


# In[ ]:


#general parameters
fs=44100 #sampling freq
Nf = 2000 #length of IR, number of freq components in double-sided spectrum
dt = 1/fs #time step
df = 1/(dt*Nf) #frequency step
freq = np.arange(0,fs/2+df,df) #frequency vector


# In[ ]:


#calculate the non-coincidence corrected signals W,X,Y,Z for different source positions
a = 0.5 #capsule directivity pattern

#choose source position
s_pos1 = [2,ma.pi/2,0] #front (0°)
s_pos2 = [2,ma.pi/2,1*ma.pi/8] # 22.5°
s_pos3 = [2,ma.pi/2,2*ma.pi/8]  #45°
s_pos4 = [2,ma.pi/2,3*ma.pi/8]# 67.5°
s_pos5 = [2,ma.pi/2,ma.pi/2] # 90°

W1,X1,Y1,Z1 = bformat(a,freq,s_pos1)
W2,X2,Y2,Z2 = bformat(a,freq,s_pos2)
W3,X3,Y3,Z3 = bformat(a,freq,s_pos3)
W4,X4,Y4,Z4 = bformat(a,freq,s_pos4)
W5,X5,Y5,Z5 = bformat(a,freq,s_pos5)

WB1 = corr_batke(a,0,W1,freq)
XB1 = corr_batke(a,1,X1,freq)
YB1 = corr_batke(a,1,Y1,freq)
ZB1 = corr_batke(a,1,Z1,freq)
WB2 = corr_batke(a,0,W2,freq)
XB2 = corr_batke(a,1,X2,freq)
YB2 = corr_batke(a,1,Y2,freq)
ZB2 = corr_batke(a,1,Z2,freq)
WB3 = corr_batke(a,0,W3,freq)
XB3 = corr_batke(a,1,X3,freq)
YB3 = corr_batke(a,1,Y3,freq)
ZB3 = corr_batke(a,1,Z3,freq)
WB4 = corr_batke(a,0,W4,freq)
XB4 = corr_batke(a,1,X4,freq)
YB4 = corr_batke(a,1,Y4,freq)
ZB4 = corr_batke(a,1,Z4,freq)
WB5 = corr_batke(a,0,W5,freq)
XB5 = corr_batke(a,1,X5,freq)


# In[ ]:


#calculate the impulse responses from the spectra and create the time signals w,x,y,z
w1 = np.real(np.fft.ifft(doubside(WB1)))
x1 = np.real(np.fft.ifft(doubside(XB1)))
y1 = np.real(np.fft.ifft(doubside(YB1)))
z1 = np.real(np.fft.ifft(doubside(ZB1)))
w2 = np.real(np.fft.ifft(doubside(WB2)))
x2 = np.real(np.fft.ifft(doubside(XB2)))
y2 = np.real(np.fft.ifft(doubside(YB2)))
z2 = np.real(np.fft.ifft(doubside(ZB2)))
w3 = np.real(np.fft.ifft(doubside(WB3)))
x3 = np.real(np.fft.ifft(doubside(XB3)))
y3 = np.real(np.fft.ifft(doubside(YB3)))
z3 = np.real(np.fft.ifft(doubside(ZB3)))
w4 = np.real(np.fft.ifft(doubside(WB4)))
x4 = np.real(np.fft.ifft(doubside(XB4)))
y4 = np.real(np.fft.ifft(doubside(YB4)))
z4 = np.real(np.fft.ifft(doubside(ZB4)))


time = np.arange(0,dt*Nf,dt)
plt.plot(time,w1,'b--')
plt.plot(time,x1,'r-')
plt.plot(time,y1,'m:')
plt.plot(time,z1,'y-.')
plt.show()


# In[ ]:


#choose a music file 
filename = "filename.wav"
data, fs = sf.read(filename, dtype='float32')


# In[ ]:


w1_conv = np.convolve(data,w1)
x1_conv = np.convolve(data,x1)
y1_conv = np.convolve(data,y1)
z1_conv = np.convolve(data,z1)


# In[ ]:


sf.write('w_filename.wav', w1_conv,fs)
sf.write('x_filename.wav', x1_conv,fs)
sf.write('y_filename.wav', y1_conv,fs)
sf.write('z_filename.wav', z1_conv,fs)

