
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


#returns the B-format signal of the mic 
#input parameters
# a: mic directivity constant
# A: amplitude of sound source
# freq: frequency vector
# s_pos: source position
def bformat(a,A,freq = [],s_pos = []):
     
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
            p_LF = A*ma.exp(-1j*k*d_LF)
            p_LB = A*ma.exp(-1j*k*d_LB)
            p_RB = A*ma.exp(-1j*k*d_RB)
            p_RF = A*ma.exp(-1j*k*d_RF)
    
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


#Measurement simulation
f = np.arange(0,100000,1) #frequency vector
h_s = 1 #distance source to absorber
h = 0.1 #distance mic to absorber
R_abs = 0.3-0.3j #reflection coefficient of the absorber

W100,X100,Y100,Z100 = bformat(0.5,1,f,[(h_s-h),np.pi/2,np.pi])  #direct incident sound
W100refl,X100refl,Y100refl,Z100refl = bformat(0.5,R_abs,f,[(h_s+h),np.pi/2,0]) #reflected sound

#non-coincidence correction
WF100 = corr_batke(0.5,0,W100,f)
XF100 = corr_batke(0.5,1,X100,f)
WF100refl = corr_batke(0.5,0,W100refl,f)
XF100refl = corr_batke(0.5,1,X100refl,f)

W100meas = WF100+WF100refl
X100meas = XF100+XF100refl

Zfrf = (WF100)/(XF100) # free field impedance
Zmeas = (W100meas)/(X100meas) # sound field impedance during material measurement

c = 340
k = 2*np.pi*f/c
Rmic = ((Zmeas/Zfrf -1)/(Zmeas/Zfrf +1)) * np.exp(1j*k*2*h)


# In[ ]:


plt.semilogx(np.real(Rmic),label='real')
plt.semilogx(np.imag(Rmic),label = 'imaginary')
plt.semilogx(abs(Rmic),label='absolute')
plt.legend()
plt.show()

