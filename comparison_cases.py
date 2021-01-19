# -*- coding: utf-8 -*-
"""
Spyder Editor

pipe conductivity with heat generation example

case 1 heat flux defined at OD and temp set at ID
case 2 temperatures at OD and ID defined.
case 3 2 different materials, inner and outer temps defined, continous
temperature distribution
case 4 2 different materials, inner and outer coolant temps and HTC defined.
"""
import numpy as np
import matplotlib.pyplot as plt
u=20e3 #mass specific volumetric heating (W/kg)
Qo = 0.1e6 #heat load on outside of pipe (W/m^2)
ri = 5e-3 #pipe inner radius in m
t = np.linspace(0.5,10,11)*1e-3 #pipe thickness in m
ro = ri+t
k = 30 #approx Eurofer conductivity W/m/K
rho=7798 #Eurofer 97 density (kg/m^3)
Ti = 300 #inner wall temp in Celsius
To = 450 #outer wall temp in Celsius
n = 50 #number of elements in 1D heat transfer evaluation
hi = 1e4 #random example HTC W/m^2K
ho = 1e4 #random example HTC W/m^2K
R12 =1e-5 #interlayer resistance m^2K/W
def T_profile_case_1(Qo,Ti,rho):
    r_var, T_res = np.zeros((len(t),n)),np.zeros((len(t),n))
    C1 = 1/(2*k)*(u*rho*ro**2+Qo/(np.pi)) #inegration constant based on outer heat flux Qo
    C2 = Ti + u*rho*ri**2/(4*k)-C1*np.log(ri) #integration constant based on inner temp
    for i in range(len(t)):
        r_var[i] = ri+t[i]*np.linspace(0,1,n)
        T_res[i] = -u*rho/(4*k)*r_var[i]**2+C1[i]*np.log(r_var[i])+C2[i]
    fig1, ax1= plt.subplots(1)
    ax1.plot((np.transpose(r_var)-ri)*1e3, np.transpose(T_res))
    ax1.set_xlabel('Pipe thickness (mm)')
    ax1.set_ylabel(r'Temperature $^{\circ}$C')
    return r_var, T_res
#results_1=T_profile_case_1(Qo, Ti,rho_a)
def T_profile_case_2(Ti,To,rho):
    r_var, T_res = np.zeros((len(t),n)),np.zeros((len(t),n))
    Q_outer, Q_inner = np.zeros((len(t))), np.zeros((len(t))),
    for i in range(len(t)):
        A_ = np.array([[np.log(ri),1],[np.log(ro[i]),1]], dtype=float)
        B_ = np.array([[Ti+u*rho*ri**2/(4*k)],[To+u*rho*ro[i]**2/(4*k)]])
        C_ = np.linalg.solve(A_,B_)
        r_var[i] = ri+t[i]*np.linspace(0,1,n)
        T_res[i] = -u*rho/(4*k)*r_var[i]**2+C_[0]*np.log(r_var[i])+C_[1]
        Q_inner[i] = -k*(2*np.pi*ri)*(-u*rho/(2*k)*ri+C_[0]/ri)
        Q_outer[i] = -k*(2*np.pi*ro[i])*(-u*rho/(2*k)*ro[i]+C_[0]/ro[i])
    fig1, ax1= plt.subplots(1)
    ax1.plot((np.transpose(r_var)-ri)*1e3, np.transpose(T_res))
    ax1.set_xlabel('Pipe thickness (mm)')
    ax1.set_ylabel(r'Temperature $^{\circ}$C')
    fig2, ax2= plt.subplots(1)
    ax2.plot(t*1e3, Q_inner/1e3, 'r-', label='Inner diameter')
    ax2.plot(t*1e3, Q_outer/1e3, 'b-', label='Outer diameter')
    ax2.legend(loc="lower right")
    ax2.set_xlabel('Pipe thickness (mm)')
    ax2.set_ylabel(r'Heat transfer kW/m')
    return r_var, T_res, Q_inner, Q_outer
#results_2=T_profile_case_2(Ti, To, rho_a)
def temperature(r, C1, C2, U, k):
    return -U*r**2/(4*k)+C1*np.log(r)+C2
def heat_flux(r, C1, U, k):
    return -k*(-U*r/(2*k)+C1/r)
def T_profile_case_3(Ti, To, U1, U2, k1, k2, ri, ro, r12):
    #set up matrices to calculate integration constants.
    A_ = np.array([[np.log(ri)  ,1 ,0          ,0],
                   [0           ,0 ,np.log(ro) ,1],
                   [-2*k1       ,0 ,2*k2       ,0],
                   [-np.log(r12),-1,np.log(r12),1]])
    B_ = np.array([[Ti+U1*ri**2/(4*k1)],
                   [To+U2*ro**2/(4*k2)],
                   [r12**2*(U2-U1)],
                   [r12**2*(U2/k2-U1/k1)/4]])
    C_ = np.linalg.solve(A_,B_)
    r_, T_=np.zeros(2*n),np.zeros(2*n)
    r1_ = np.linspace(ri,r12,n)
    r2_ = np.linspace(r12,ro,n)
    r_[0:n]=r1_
    r_[n::]=r2_
    T_[0:n]=temperature(r1_,C_[0],C_[1],U1, k1)
    T_[n::]=temperature(r2_,C_[2],C_[3],U2, k2)
    fig1, ax1= plt.subplots(1)
    ax1.plot(r_*1e3,T_)
    ax1.set_xlabel('Radius (mm)')
    ax1.set_ylabel(r'Temperature $^{\circ}$C')
    Q_inner = heat_flux(ri,C_[0],U1,k1)*2*np.pi*ri
    Q_outer = heat_flux(ro,C_[2],U2,k2)*2*np.pi*ro
    Q_internal = np.pi*(U2*(ro**2-r12**2)+U1*(r12**2-ri**2))
    print("Q_inner={}W/m \n".format(Q_inner) +
          "Q_outer={}W/m \n".format(Q_outer) +
          "Internal heat generarion = {}W/m".format(Q_internal)
        )
    return r_, T_
#T_profile_case_3(Ti,To,u*rho,u*rho,k,k*5,ri,ri+3e-3,ri+1e-3)
def T_profile_case_4(Ti_inf, To_inf, h1, h2, U1, U2, k1, k2, ri, ro, r12):
    #set up matrices to calculate integration constants.
    A_ = np.array([[k1/(h1*ri)-np.log(ri),-1 ,0                    ,0],
                   [0                    ,0  ,k2/(h2*ro)+np.log(ro),1],
                   [-2*k1                ,0  ,2*k2                 ,0],
                   [-np.log(r12)         ,-1 ,np.log(r12)          ,1]])
    B_ = np.array([[U1*ri/2*(1/h1-ri/(2*k1))-Ti_inf],
                   [U2*ro/2*(1/h2+ro/(2*k2))+To_inf],
                   [r12**2*(U2-U1)],
                   [r12**2*(U2/k2-U1/k1)/4]])
    C_ = np.linalg.solve(A_,B_)
    r_, T_=np.zeros(2*n),np.zeros(2*n)
    r1_ = np.linspace(ri,r12,n)
    r2_ = np.linspace(r12,ro,n)
    r_[0:n]=r1_
    r_[n::]=r2_
    T_[0:n]=temperature(r1_,C_[0],C_[1],U1, k1)
    T_[n::]=temperature(r2_,C_[2],C_[3],U2, k2)
    fig1, ax1= plt.subplots(1)
    ax1.plot(r_*1e3,T_)
    ax1.plot(ri*1e3+np.array([-0.1,0,0]),np.array([Ti_inf,Ti_inf,float(T_[0])]))
    ax1.plot(ro*1e3+np.array([0,0,0.1]),np.array([float(T_[-1]),To_inf,To_inf]))
    ax1.set_xlabel('Radius (mm)')
    ax1.set_ylabel(r'Temperature $^{\circ}$C')
    Q_inner = heat_flux(ri,C_[0],U1,k1)*2*np.pi*ri
    Q_outer = heat_flux(ro,C_[2],U2,k2)*2*np.pi*ro
    Q_internal = np.pi*(U2*(ro**2-r12**2)+U1*(r12**2-ri**2))
    print("Q_inner={}W/m \n".format(Q_inner) +
          "Q_outer={}W/m \n".format(Q_outer) +
          "Internal heat generarion = {}W/m \n".format(Q_internal) +
          "Q_inner-Q_outer = {} W/m".format(Q_inner-Q_outer)     
        )
    return r_, T_
#results4=T_profile_case_4(Ti,To,hi,ho,u*rho,u*rho*10,k,k*5,ri,ri+2e-3,ri+0.5e-3)
def T_profile_case_5(Ti_inf, To_inf, h1, h2, U1, U2, k1, k2, ri, ro, r12, R12):
    #set up matrices to calculate integration constants.
    A_ = np.array([[k1/(h1*ri)-np.log(ri) ,-1 ,0                    ,0],
                   [0                     ,0  ,k2/(h2*ro)+np.log(ro),1],
                   [-2*k1                 ,0  ,2*k2                 ,0],
                   [np.log(r12)+R12*k1/r12,1  ,-np.log(r12)         ,-1]])
    B_ = np.array([[U1*ri/2*(1/h1-ri/(2*k1))-Ti_inf],
                   [U2*ro/2*(1/h2+ro/(2*k2))+To_inf],
                   [r12**2*(U2-U1)],
                   [r12/2*(U1*(R12+r12/(2*k1))-U2*r12/(2*k2))]])
    C_ = np.linalg.solve(A_,B_)
    print(C_)
    r_, T_=np.zeros(2*n),np.zeros(2*n)
    r1_ = np.linspace(ri,r12,n)
    r2_ = np.linspace(r12,ro,n)
    r_[0:n]=r1_
    r_[n::]=r2_
    T_[0:n]=temperature(r1_,C_[0],C_[1],U1, k1)
    T_[n::]=temperature(r2_,C_[2],C_[3],U2, k2)
    fig1, ax1= plt.subplots(1)
    ax1.plot(r_*1e3,T_)
    ax1.plot(ri*1e3+np.array([-0.1,0,0]),np.array([Ti_inf,Ti_inf,float(T_[0])]))
    ax1.plot(ro*1e3+np.array([0,0,0.1]),np.array([float(T_[-1]),To_inf,To_inf]))
    ax1.set_xlabel('Radius (mm)')
    ax1.set_ylabel(r'Temperature $^{\circ}$C')
    Q_inner = heat_flux(ri,C_[0],U1,k1)*2*np.pi*ri
    Q_outer = heat_flux(ro,C_[2],U2,k2)*2*np.pi*ro
    Q_internal = np.pi*(U2*(ro**2-r12**2)+U1*(r12**2-ri**2))
    print("Q_inner={}W/m \n".format(Q_inner) +
          "Q_outer={}W/m \n".format(Q_outer) +
          "Internal heat generarion = {}W/m \n".format(Q_internal) +
          "Q_inner-Q_outer = {} W/m".format(Q_inner-Q_outer)     
        )
    return r_, T_
results5=T_profile_case_5(Ti,To,hi,ho,0,1e8,k*10,k,ri,ri+4e-3,ri+0.5e-3, R12)