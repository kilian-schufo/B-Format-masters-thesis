
# coding: utf-8

# In[ ]:


#load modules
import cmath as ma
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().magic('matplotlib inline')
from mpl_toolkits.mplot3d import Axes3D


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
#gives always the smallest angle between the vectors! It is okay here because the cosine is symmetrical to pi
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


#returns the B-format signal of the mic for a given freq at every angle

def bformat(f,a):
     
    w = 2*ma.pi*f
    c = 340
    k = w/c
    
    #memory allocation output signals
    s_LF = np.zeros((200,200),dtype=np.complex_)
    s_LB = np.zeros((200,200),dtype=np.complex_)
    s_RB = np.zeros((200,200),dtype=np.complex_)
    s_RF = np.zeros((200,200),dtype=np.complex_)

    #sound source position
    s_phi = np.linspace(0,2*np.pi,num=200) #from 0 to 2pi in 200 steps
    s_teta = np.linspace(0,np.pi,num=200)  # from 0 to pi in 200 steps


    for i in range(200):
    
        for ii in range(200):
    
            s_pos = ([10, s_teta[ii], s_phi[i]])


            #capsule positions
            R = 0.0147 # radius of the tetrahedron in meters according to Gerzon
            tilt = ma.atan(1/ma.sqrt(2))  # tilt of the capsules in rad according to Farrar
    
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
        
            #pressure at the capsules due to the source at s_pos, plane waves assumed
            p_LF = ma.exp(-1j*k*d_LF)
            p_LB = ma.exp(-1j*k*d_LB)
            p_RB = ma.exp(-1j*k*d_RB)
            p_RF = ma.exp(-1j*k*d_RF)
    
            #signals of the capsules, (increasing teta along the rows, increasing phi along the columns)
            s_LF[ii,i] = (a + a*ma.cos(angLF))*p_LF
            s_LB[ii,i] = (a + a*ma.cos(angLB))*p_LB
            s_RB[ii,i] = (a + a*ma.cos(angRB))*p_RB
            s_RF[ii,i] = (a + a*ma.cos(angRF))*p_RF
            
    #B-format signals
    W = s_LF + s_RB + s_RF + s_LB #omni
    X = s_LF - s_RB + s_RF - s_LB #f.o.e. forward (to positive x)
    Y = s_LF - s_RB - s_RF + s_LB #f.o.e. leftward (to positive y)
    Z = s_LF - s_LB + s_RB - s_RF #f.o.e. upward (to positive z)
    
    return W,X,Y,Z


# In[ ]:


#non-coincidence correction according to Gerzon as introduced by Batke, W attenuated by 3dB
def corr_batke(B_format_matrix,order,f,a):

    R = 0.0147
    w = 2*ma.pi*f
    c = 340
    k = w/c
    xx = k*R

    # Spherical Bessel functions of first kind
    j_0 = ma.sin(xx)/(xx)
    j_0_diff = (xx*ma.cos(xx) - ma.sin(xx))/((xx)**2)
    j_1 = ma.sin(xx)/((xx)**2) - ma.cos(xx)/(xx)
    j_1_diff = ((xx**2 - 2)*ma.sin(xx) + 2*xx*ma.cos(xx))/(xx**3)

    V_0 = 1/(a*j_0 - 1j*(1-a)*j_0_diff)
    V_1 = 1/(a*j_1 - 1j*(1-a)*j_1_diff)
    
    if order == 0:
        return B_format_matrix*V_0*(1/ma.sqrt(4*ma.pi))*(1/np.sqrt(2))
    if order == 1:
        return B_format_matrix*V_1*(1/ma.sqrt(4*ma.pi))*(-1j)


# In[ ]:


#non-coincidence correction according to Faller et al., W attenuated by 3 dB
def corr_faller(B_format_matrix,f,name):
    
    s_phi = np.linspace(0,2*np.pi,num=200) #from 0 to 2pi in 200 steps
    s_teta = np.linspace(0,np.pi,num=200) #from 0 to pi in 200 steps
    
    if name == 'W':
        num = np.sum(np.conjugate(B_format_matrix))
        denom = np.sum(np.multiply(B_format_matrix,np.conjugate(B_format_matrix)))
        return B_format_matrix*(num/denom)*(1/np.sqrt(2))
    
    if name == 'X':
        f_o_e_x = np.zeros((200,200),dtype=np.complex_)

        for i in range(200):
            for ii in range(200):
                s_pos = ([10, s_teta[ii], s_phi[i]])
                m_pos = ([0, ma.pi/2, 0]) #mic looking towards positive x-axis
                ang = angle(*s_pos,*m_pos)
                f_o_e_x[ii,i] = ma.cos(ang) #optimal f.o.e. pattern along x-axis
        
        num = np.sum(np.multiply(f_o_e_x,np.conjugate(B_format_matrix)))
        denom = np.sum(np.multiply(B_format_matrix,np.conjugate(B_format_matrix)))
        return B_format_matrix*(num/denom)
    
    if name == 'Y':
        f_o_e_y = np.zeros((200,200),dtype=np.complex_)
        for i in range(200):
            for ii in range(200):
                s_pos = ([10, s_teta[ii], s_phi[i]])
                m_pos = ([0, ma.pi/2, ma.pi/2])
                ang = angle(*s_pos,*m_pos)
                f_o_e_y[ii,i] = ma.cos(ang)
        num = np.sum(np.multiply(f_o_e_y,np.conjugate(B_format_matrix)))
        denom = np.sum(np.multiply(B_format_matrix,np.conjugate(B_format_matrix)))
        return B_format_matrix*(num/denom)
    
    
    if name == 'Z':
        f_o_e_z = np.zeros((200,200),dtype=np.complex_)
        for i in range(200):
            for ii in range(200):
                s_pos = ([10, s_teta[ii], s_phi[i]])
                m_pos = ([0, 0, 0])
                ang = angle(*s_pos,*m_pos)
                f_o_e_z[ii,i] = ma.cos(ang)
        num = np.sum(np.multiply(f_o_e_z,np.conjugate(B_format_matrix)))
        denom = np.sum(np.multiply(B_format_matrix,np.conjugate(B_format_matrix)))
        return B_format_matrix*(num/denom)


# In[ ]:


#unfiltered B-format signals
a = 0.5
W200,X200,Y200,Z200 = bformat(200,a)
W800,X800,Y800,Z800 = bformat(800,a)
W4000,X4000,Y4000,Z4000 = bformat(4000,a)
W8000,X8000,Y8000,Z8000 = bformat(8000,a)
W10000,X10000,Y10000,Z10000 = bformat(10000,a)
W15000,X15000,Y15000,Z15000 = bformat(15000,a)


# In[ ]:


#filterd with Filter suggested by Gerzon, W signal -3 dB
WB200 = corr_batke(W200,0,200,a)
WB800 = corr_batke(W800,0,800,a)
WB4000 = corr_batke(W4000,0,4000,a)
WB8000 = corr_batke(W8000,0,8000,a)
WB10000 = corr_batke(W10000,0,10000,a)
WB15000 = corr_batke(W15000,0,15000,a)

XB200 = corr_batke(X200,1,200,a)
XB800 = corr_batke(X800,1,800,a)
XB4000 = corr_batke(X4000,1,4000,a)
XB8000 = corr_batke(X8000,1,8000,a)
XB10000 = corr_batke(X10000,1,10000,a)
XB15000 = corr_batke(X15000,1,15000,a)

YB200 = corr_batke(Y200,1,200,a)
YB800 = corr_batke(Y800,1,800,a)
YB4000 = corr_batke(Y4000,1,4000,a)
YB8000 = corr_batke(Y8000,1,8000,a)
YB10000 = corr_batke(Y10000,1,10000,a)
YB15000 = corr_batke(Y15000,1,15000,a)

ZB200 = corr_batke(Z200,1,200,a)
ZB800 = corr_batke(Z800,1,800,a)
ZB4000 = corr_batke(Z4000,1,4000,a)
ZB8000 = corr_batke(Z8000,1,8000,a)
ZB10000 = corr_batke(Z10000,1,10000,a)
ZB15000 = corr_batke(Z15000,1,15000,a)


# In[ ]:


#filtered as suggested by Faller et al., W signal -3 dB
WF200 = corr_faller(W200,200,'W')
WF800 = corr_faller(W800,800,'W')
WF4000 = corr_faller(W4000,4000,'W')
WF8000 = corr_faller(W8000,8000,'W')
WF10000 = corr_faller(W10000,10000,'W')
WF15000 = corr_faller(W15000,15000,'W')

XF200 = corr_faller(X200,200,'X')
XF800 = corr_faller(X800,800,'X')
XF4000 = corr_faller(X4000,4000,'X')
XF8000 = corr_faller(X8000,8000,'X')
XF10000 = corr_faller(X10000,10000,'X')
XF15000 = corr_faller(X15000,15000,'X')

YF200 = corr_faller(Y200,200,'Y')
YF800 = corr_faller(Y800,800,'Y')
YF4000 = corr_faller(Y4000,4000,'Y')
YF8000 = corr_faller(Y8000,8000,'Y')
YF10000 = corr_faller(Y10000,10000,'Y')
YF15000 = corr_faller(Y15000,15000,'Y')

ZF200 = corr_faller(Z200,200,'Z')
ZF800 = corr_faller(Z800,800,'Z')
ZF4000 = corr_faller(Z4000,4000,'Z')
ZF8000 = corr_faller(Z8000,8000,'Z')
ZF10000 = corr_faller(Z10000,10000,'Z')
ZF15000 = corr_faller(Z15000,15000,'Z')

