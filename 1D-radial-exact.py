# -*- coding: utf-8 -*-
"""
Attempt to creat a generalised 1-D code for heat transfer through concentric
cylinders with internal heat generation. Boundary condition support:
    1. Two temperatures (inner/outer) defined.
    2. Two (inner and outer) sets of bulk coolant temperatures and HTCs
    defined.
    3. A combination of the above.
    2. A temperature or bulk temperature and HTC and a heat flux (at inner/
    outer boundaries).
Will use "layer" classes with solid and fluid subclasses.
Solid layers will be defined by:
    Thickness, conductivity, bulk heating, outer thermal resistance.
    Outer thermal resistance ignored if:
        -is ultimate outer edge.
        -is not between two fluid layers.
Fluid layers will be defined by:
    Thickness, bulk temperature, inner HTC and outer HTC.
Thermal resistance can be added after a solid layer but MUST be followed by a
solid layer.
Radial outward build configuration.
    First layer must be solid.
    Solid layers can be followed by a:
        -thermal resistance,
        -solid layer,
        -fluid layer.
    Thermal resistance must be follwed by:
        -Solid layer
    Fluid layer must be followed by:
        -Solid layer.
    Final layer must be a solid layer.
    
@author: bruce
"""
import numpy as np
from matplotlib import pyplot as plt
class layer:
    def __init__(self,  t):
        if t<=0:
            raise ValueError('please enter positive non-zero layer thickness for {}'.format(self))
        self.t = t #layer thickness [m]
class solid_layer(layer):
    def __init__(self, t, k, V_heat = 0, R_out = 0):
        super().__init__(t)
        self.layer_type = 'solid'
        self.k = k #conductivity in solid [W/kg/K]
        self.V_heat = V_heat #volumetric heating [W/m^3] 
        self.R_out = R_out #thermal resistance at outer edge [m^2K/W]
class fluid_layer(layer):
    def __init__(self, t, T_inf, h_in, h_out):
        super().__init__(t)
        self.layer_type = 'fluid'
        self.T_inf = T_inf #bulk coolant temperature [degC]
        if (h_in <= 0 or h_out <= 0):
            raise ValueError('Please enter positive, non-zero HTC values for {}'.format(self))
        self.h_in = h_in #HTC at inner radial surface [W/m^2K]
        self.h_out = h_out #HTC at outer radial surface [W/m^2K]
class boundary_condition:
    def __init__(self, BC_type, BC_side, **kwargs):
        kword_dict = {'temperature':{'T_inf':1,'R':0},
                      'fluid':{'T_inf':1,'h':1},
                      'heat flux':{'q':1}}
        if BC_type not in kword_dict:
            raise TypeError('Please enter allowed "BC_type" for {} \n'.format(self)+
                            'from list:' + str(list(kword_dict)))  #not sure if this can really be considered a "TypeError".         
        if BC_side == 'inner' or BC_side =='outer':
            self.BC_side = BC_side
        else:
            raise ValueError('Invalid input for positional arguemrnt BC_side for {}:\n'.format(self)+
                             'must be str "inner" or "outer" to indicate inner or outer BC respectively.')
        self.BC_type = BC_type
        for (key, val) in kwargs.items():
            if key not in kword_dict[BC_type]:
                print('Redundant keyword arguement "{}" for BC_type "{}" for {}'.format(key,BC_type,self))
            else:
                setattr(self, key, val)
        for (key, val) in kword_dict[BC_type].items():
            if val == 1 and (key not in kwargs):
                raise NameError('Missing required keyword arguement "{}" for BC_type "{}" for {}'.format(key,BC_type,self))
class concentric_cylinder:
    def __init__(self, rad_inner, layers, BC_1, BC_2):
        if rad_inner<=0:
            raise ValueError('rad_inner must be non-zero and positive.')
        if layers[0].layer_type == 'fluid':
            raise RuntimeError('First layer cannot be a "fluid_layer" object.\n'+
                              'Use fluid inner boundary condition instead.')
        if layers[-1].layer_type == 'fluid':
            raise RuntimeError('Final layer cannot be a "fluid_layer" object.\n'+
                              'Use fluid outer boundary condition instead.')
        if (BC_1.BC_type == 'heat flux') and (BC_2.BC_type == 'heat flux'):
            raise RuntimeError('Cannot have two heat flux boundary conditions.')
        if (BC_1.BC_side==BC_2.BC_side) and ((BC_1.BC_type!='heat flux') and (BC_2.BC_type!='heat flux')):
            raise RuntimeError('Cannot have two BCs at same boundary unless 1 is type "heat flux".')
        radii = [rad_inner]
        solid_map=[]
        fluid_map=[]         
        for i in range(len(layers)):
            if layers[i].layer_type=='fluid' and layers[i+1].layer_type=='fluid':
                raise RuntimeError('Cannot have two consecutive fluid layers\n'+
                                   '({} and {} in {})'.format(layers[i],layers[i+1],self))
            elif layers[i].layer_type=='solid':
                solid_map.append(1)
                fluid_map.append(0)
            elif layers[i].layer_type=='fluid':
                solid_map.append(0)
                fluid_map.append(1)
            radii.append(radii[i]+layers[i].t)
        self.radii = radii
        self.solid_map = solid_map
        self.fluid_map = fluid_map
        self.layers = layers
        self.BC = [BC_1, BC_2]
        self.BC_in = []
        self.BC_out = []
        for BC_ in self.BC:
            if BC_.BC_side == 'inner':
                self.BC_in.append(BC_)
            else:
                self.BC_out.append(BC_)
    def C(self):
        
        """
        Returns integration constants to fully define temperature profile,
        two are needed to define the temperature profile in each solid layer.
        The system of equations are set up using 2 boundary conditions,
        A heat continuity equation for each solid-solid boundary.
        A temperature equation for each boundary.
        -------
        """
        def A_for_T_bound(sgn,r,k,R):
            return np.array([[sgn*k*R/r-np.log(r), -1]])
        def B_for_T_bound(sgn,r,k,R,V,T_inf):
            return V*r/2*(sgn*R-r/(2*k)) - T_inf
        #initialise matrices
        N=self.solid_map.count(1)
        A_, B_ = np.zeros((2*N,2*N)), np.zeros((2*N,1))
        j=0 # row tracker
        #apply boundary conditions:
        for BC_ in self.BC:
            if BC_.BC_side == 'inner':
                n, sgn=0, 1 # first layer index, sign for heat resistance term
                r = self.radii[0]
                V = self.layers[0].V_heat
                k = self.layers[0].k
            else:
                n, sgn=N-1,-1 #last layer index, sign for heat resistance term
                r = self.radii[-1]
                V = self.layers[-1].V_heat
                k = self.layers[-1].k
            if BC_.BC_type!='heat flux':
                if BC_.BC_type=='temperature':
                    R= BC_.R
                else:
                    R= 1/BC_.h
                A_[j,2*n:2*n+2]=A_for_T_bound(sgn,r,k,R)
                B_[j,0] = B_for_T_bound(sgn,r,k,R,V,BC_.T_inf)
                j+=1
            else:
                A_[j,2*n:2*n+2]=np.array([[1, 0]])
                B_[j,0] = -r/k*(BC_.q - V*r/2)
                j+=1    
        #apply layer temp and heat continuity
        for i in range(len(self.layers)-1):
            i_sol = sum(self.solid_map[0:i])
            if self.layers[i].layer_type == self.layers[i+1].layer_type:
                #heat continuity
                r=self.radii[i+1]
                k0=self.layers[i].k
                k1=self.layers[i+1].k
                V0=self.layers[i].V_heat
                V1=self.layers[i+1].V_heat
                R=self.layers[i].R_out
                A_[j,2*i_sol:2*i_sol+4]=np.array([[-k0,0,k1,0]])
                B_[j,0]=r**2/2*(V1-V0)
                j+=1
                #temp continuity
                A_[j,2*i_sol:2*i_sol+4]=np.array([[np.log(r)+R*k0/r, 1, -np.log(r), -1]])
                B_[j,0]=r/2*(V0*(R+r/(2*k0))-V1*r/(2*k1))
                j+=1
            elif self.layers[i].layer_type == 'solid':
                #define based on solid-fluid transition
                r=self.radii[i+1] 
                k=self.layers[i].k
                V=self.layers[i].V_heat
                R=1/(self.layers[i+1].h_in)
                T_inf = self.layers[i+1].T_inf
                A_[j,2*i_sol:2*i_sol+2]=A_for_T_bound(-1,r,k,R)
                B_[j,0] = B_for_T_bound(-1,r,k,R,V,T_inf)
                j+=1
                #second boundary condition (fluid-solid)
                r=self.radii[i+2]
                k=self.layers[i+2].k
                V=self.layers[i+2].V_heat
                R=1/(self.layers[i+1].h_out)
                T_inf = self.layers[i+1].T_inf
                A_[j,2*(i_sol+1):2*(i_sol+1)+2]=A_for_T_bound(1,r,k,R)
                B_[j,0] = B_for_T_bound(1,r,k,R,V,T_inf)
                j+=1
            else:
                pass
            # else:
            #     #if fluid-solid boundary add to j?
            #     r=self.radii[i+1]
            #     k=self.layers[i+1].k
            #     V=self.layers[i+1].V_heat
            #     R=1/(self.layers[i].h_out)
            #     T_inf = self.layers[i].T_inf
            #     A_[j,2*i:2*i+2]=A_for_T_bound(1,r,k,R)
            #     B_[j,0] = B_for_T_bound(1,r,k,R,V,T_inf)
            #     j+=1
        self.A_=A_
        self.B_=B_
        C_ = np.linalg.lstsq(A_,B_,rcond=None)[0]
        self.C_=C_
    def layer_index_for_r(self,rad):
        """

        Parameters
        ----------
        rad : int or float
            radius in m

        Returns
        -------
        layer : layer object which covers that radial position
        if at a boundary returns the outer layer or "inner" and "outer" to
        indicate being in the "inner" or "outer" boundary condition regime.
        """
        if rad < self.radii[0]:
            return 'inner'
        elif rad >= self.radii[-1]:
            return 'outer'
        else: 
            for i in range(len(self.layers)):
                if rad >= self.radii[i] and rad< self.radii[i+1]:
                    return i            
    def T(self,rad):
        layer_i = self.layer_index_for_r(rad)
        if layer_i == 'inner':
            if not self.BC_in:
                return self.T_solid(self.radii[0],self.layers[0],0)[0]
            else:
                for BC_ in self.BC_in:
                    if BC_.BC_type!='heat flux':
                        return BC_.T_inf
                    else:
                        return self.T_solid(self.radii[0],self.layers[0],0)[0]
            return #something if inner temp or fluid BC exists do solid with radii[0] if not
        elif layer_i == 'outer':
            if not self.BC_out:
                return self.T_solid(self.radii[-1],self.layers[-1],len(self.layers)-1)[0]
            else:
                for BC_ in self.BC_out:
                    if BC_.BC_type!='heat flux':
                        return BC_.T_inf
                    else:
                        return self.T_solid(self.radii[-1],self.layers[-1],len(self.layers)-1)[0]
        else: 
            layer = self.layers[layer_i]
            if layer.layer_type == 'fluid':
                return layer.T_inf
        
            elif layer.layer_type=='solid':
                return self.T_solid(rad,layer,layer_i)[0]
    def q(self,rad):
        #calculating heat flux at a radial position
        if not hasattr(self,'C_'):
            self.C()
        layer_i = self.layer_index_for_r(rad)
        if layer_i == 'outer' or layer_i == 'inner':
            return np.nan
        elif self.layers[layer_i].layer_type == 'fluid':
            return np.nan
        else:
            layer = self.layers[layer_i]
            V_heat, k = layer.V_heat, layer.k
            solid_layer_i = sum(self.solid_map[0:layer_i])
            CA = self.C_[2*solid_layer_i][0]
            return -(k/rad*CA-V_heat*rad/2)
        
        return
    def Q(self,rad):
        #calculating heat transfer at a radial position
        return self.q(rad)*2*np.pi*rad
        
        return
    def T_solid(self,rad,layer,layer_i):
        if not hasattr(self,'C_'):
            self.C()
        V_heat, k = layer.V_heat, layer.k
        solid_layer_i = sum(self.solid_map[0:layer_i])
        CA, CB = self.C_[2*solid_layer_i],self.C_[2*solid_layer_i+1]
        return -V_heat/(4*k)*rad**2+CA*np.log(rad)+CB
    def plot_T(self, dpoints=100, plotQ=0, plotq=0, bound_labs=0):
        """
        Plots the temperature profile through the wall thickness.
        Parameters
        ----------
        dpoints : int
            number of datapoints to plot through whole wall thickness.
        plotQ : boolean (0,1, True, False) optional
            If True plots heat transfer through solid layers, else does not.
            The default is 0.
        plotq : boolean (0,1, True, False) optional
            If True plots heat flux through solid layers, else does not.
            The default is 0.
        layer_labs : boolean (0,1, True, False) optional
            If true labels layers. The default is 0.
        Returns
        -------
        figure

        """
        T=[]
        rads = np.linspace(self.radii[0]*0.95,self.radii[-1]*1.05,dpoints)
        plotnum=1
        if plotQ==True:
            plotnum+=1
            Q=[]
        if plotq==True:
            plotnum+=1
            q=[]
        for r in rads:
            T.append(self.T(r))
            if plotQ==True:
                Q.append(self.Q(r))
            if plotq==True:
                q.append(self.q(r))
        fig=plt.figure()
        ax1 = fig.add_subplot(111)
        if plotQ==True:
            Qax = ax1.twinx()
            Qax.set_ylabel(r'Heat transfer kW/m',color='b')
            Q_plot = Qax.plot(rads*1e3,np.array(Q)/1e3,'b-', label='Heat Transfer')
        if plotq==True:
            qax = ax1.twinx()
            qax.set_ylabel(r'Heat flux MW/m$^2$',color='m')
            q_plot = qax.plot(rads*1e3,np.array(q)/1e6,'m-', label='Heat Flux')
        if bound_labs==1:
            ax1.plot([self.radii[0]*1e3,self.radii[0]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            ax1.plot([self.radii[-1]*1e3,self.radii[-1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            for i in range(len(self.layers)-1):
                if self.layers[i].layer_type==self.layers[i+1].layer_type:
                    ax1.plot([self.radii[i+1]*1e3,self.radii[i+1]*1e3],[min(T),max(T)],'k-.',linewidth=0.5)
                else:
                    ax1.plot([self.radii[i+1]*1e3,self.radii[i+1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
        ax1.set_xlabel('Radius [mm]')
        ax1.set_ylabel(r'Temperature $^{\circ}$C', color='r')
        temp_plot = ax1.plot(rads*1e3,T,'r-', label='Temperature')
        if plotq == 1 and plotQ==1:
            qax.spines['right'].set_position(('outward',60))
            
        return fig
        
a = solid_layer(1.5e-3,30,10e7,1e-5)
b = fluid_layer(1e-3,360,1e4,1e4)
c = solid_layer(0.4e-3,10,0,1e-5)
BC1 = boundary_condition('temperature', 'outer',T_inf=450, R=1e-6)
BC2 = boundary_condition('fluid', 'inner', T_inf=450, h=1e4)
BC3 = boundary_condition('fluid', 'outer', T_inf=450, h=1e4)
BCHFout = boundary_condition('heat flux', 'outer',q=-1e6)
BCHFin = boundary_condition('heat flux', 'inner',q=-1e6)
A = concentric_cylinder(5e-3,[c,a,c,b,c,a,c,c,a,a,b,a],BC3,BC2)
B = concentric_cylinder(5e-3,[c,a,c],BC3,BCHFout)
