# -*- coding: utf-8 -*-
"""
Attempt to create a generalised 1-D code for heat transfer through concentric
cylinders with internal heat generation.
The radial_heat_transfer_1d class is used to assemble and solve the system of
equations. It takes,
-the inner radius,
-a list containing layer objects,
-2 boundary conditions.

Boundary condition (boundary_condition class) support:
    1. Temperature with optional thermal resistance at the interface.
    2. Fluid BC with bulk temperature and heat transfer coefficient defined.
    3. Defined heat flux defined at the boundary.
    -These BCs (boundary conditions) are set at 'inner' or 'outer' location.
    -If two BCs of type 1 and 2 are selected then they cannot be at the same
    location.
    -Two heat flux conditions cannot be defined at once (this overconstrains
    the system for set internal heat generation).
    -Exactly 2 boundary conditions must be provided.

To build up the layer list 2 layer classes are provided:
    -Solid layers
    -Fluid layers
Solid layers have a defined thickness and conductivity with optional internal
heat generation and outer layer bound thermal resistance.
Fluid layers have defined bulk temperature and inner and outer HTC.

They can be built up as follows:
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
    
The layer properties are then arranged into a system of simultaneous equations
to solve the integration constants for all the solid layers.
After these constants have been calculated the temperature, heatflux or
heat transfer at any radial position can be trivially calculated.
@author: bruce
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
class layer:
    def __init__(self,  t, label):
        if t<=0:
            raise ValueError('please enter positive non-zero layer thickness for {}'.format(self))
        self.t = t #layer thickness [m]
        self.label=label
class solid_layer(layer):
    def __init__(self, t, k, V_heat = 0, R_out = 0, label=None):
        super().__init__(t,label)
        self.layer_type = 'solid'
        self.k = k #conductivity in solid [W/kg/K]
        self.V_heat = V_heat #volumetric heating [W/m^3] 
        self.R_out = R_out #thermal resistance at outer edge [m^2K/W]
class fluid_layer(layer):
    def __init__(self, t, T_inf, h_in, h_out, label=None):
        super().__init__(t, label)
        self.layer_type = 'fluid'
        self.T_inf = T_inf #bulk coolant temperature [degC]
        if (h_in <= 0 or h_out <= 0):
            raise ValueError('Please enter positive, non-zero HTC values for {}'.format(self))
        self.h_in = h_in #HTC at inner radial surface [W/m^2K]
        self.h_out = h_out #HTC at outer radial surface [W/m^2K]
class boundary_condition:
    def __init__(self, BC_type, BC_side, **kwargs):
        kword_dict = {'temperature':{'T_inf':1,'R':0,'label':0},
                      'fluid':{'T_inf':1,'h':1,'label':0},
                      'heat flux':{'q':1,'label':0}}
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
        if not hasattr(self,'label'):
            self.label=None
class radial_heat_transfer_1d:
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
    def plot_T(self, dpoints=100, plotQ=1, plotq=1, show_bounds=1, Q_labels=1):
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
        show_bounds : boolean (0,1, True, False) optional
            If true labels layers. Draw layer boundaries.
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
            Qax.set_ylabel(r'Heat transfer W/m',color='b')
            Qax.plot(rads*1e3,Q,'b-', label='Heat Transfer')
            Qax.ticklabel_format(style='sci',axis='y', scilimits=(-2,2))
        if plotq==True:
            qax = ax1.twinx()
            qax.set_ylabel(r'Heat flux W/m$^2$',color='m')
            qax.plot(rads*1e3,q,'m-', label='Heat Flux')
            qax.ticklabel_format(style='sci',axis='y', scilimits=(0,0))
        if show_bounds==1:
            ax1.plot([self.radii[0]*1e3,self.radii[0]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            ax1.plot([self.radii[-1]*1e3,self.radii[-1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            for i in range(len(self.layers)-1):
                if self.layers[i].layer_type==self.layers[i+1].layer_type:
                    ax1.plot([self.radii[i+1]*1e3,self.radii[i+1]*1e3],[min(T),max(T)],'k-.',linewidth=0.5)
                else:
                    ax1.plot([self.radii[i+1]*1e3,self.radii[i+1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
        ax1.set_xlabel('Radius [mm]')
        ax1.set_ylabel(r'Temperature $^{\circ}$C', color='r')
        ax1.plot(rads*1e3,T,'r-', label='Temperature')
        if plotq == 1 and plotQ==1:
            #offset axis to stop overlap
            qax.spines['right'].set_position(('outward',60))
            qax.get_yaxis().get_offset_text().set_visible(False)
            qax_max = abs(max(max(qax.get_yticks()),-min(qax.get_yticks())))
            expo = np.floor(np.log10(qax_max)).astype(int)
            qax.annotate('1e{}'.format(expo),xy=(1,1),
                         xycoords='axes fraction',
                         horizontalalignment = 'right',
                         verticalalignment='bottom',
                         xytext=(60,0),
                         textcoords='offset points')
            #align zeros if present in data
            if np.nanmax(q)>=0 and np.nanmin(q)<=0:
                q_range = np.nanmax(q)-np.nanmin(q)
                Q_range = np.nanmax(Q)-np.nanmin(Q)
                max_factor = max(np.nanmax(Q)/Q_range, np.nanmax(q)/q_range)
                min_factor = min(np.nanmin(Q)/Q_range, np.nanmin(q)/q_range)
                qax.set_ylim(min_factor*q_range, max_factor*q_range)
                Qax.set_ylim(min_factor*Q_range, max_factor*Q_range)
        if Q_labels==1:
            """
            Annotate the heat transfer at inner and outer surfaces, the nuclear
            heating in each solid layer and the net heat transfer in the fluid
            layers.
            """
            ycenter=sum(ax1.get_ylim())/2
            ax1.annotate(r'$Q_{i,net}$='+'{:.3e}'.format(-self.Q(self.radii[0]))
                         +' W/m',
                         xy=(0.05,0.5),
                         rotation='vertical',
                         xycoords='axes fraction',
                         verticalalignment='center',
                         horizontalalignment='center')
            ax1.annotate(r'$Q_{o,net}$='+'{:.3e}'.format(self.Q(self.radii[-1]*(1-1e-10)))
                         +' W/m',
                         xy=(0.95,0.5),
                         rotation='vertical',
                         xycoords='axes fraction',
                         verticalalignment='center',
                         horizontalalignment='center')
            i=0
            for layer in self.layers:
                if layer.layer_type=='solid':
                    Qvol = np.pi*(self.radii[i+1]**2-self.radii[i]**2)*layer.V_heat
                    ax1.annotate(r'$Q_{nuc}$='+'{:.3e}'.format(Qvol)
                         +' W/m',
                         xy=((sum(self.radii[i:i+2])*1e3/2),
                             ycenter),
                         rotation='vertical',
                         xycoords='data',
                         verticalalignment='center',
                         horizontalalignment='center',
                         )
                elif layer.layer_type=='fluid':
                    Qnet = self.Q(self.radii[i]*(1-1e-10))-self.Q(self.radii[i+1])
                    ax1.annotate(r'$Q_{net,fluid}$='+'{:.3e}'.format(Qnet)
                         +' W/m',
                         xy=((sum(self.radii[i:i+2])*1e3/2),
                             ycenter),
                         rotation='vertical',
                         xycoords='data',
                         verticalalignment='center',
                         horizontalalignment='center',
                         )
                i+=1 
        fig.tight_layout()
        return fig
    def results(self):
        """
        Returns a dataframe object with summarising all the inputs and main
        temperature results for easy export to a variety of formats.
        """
        rows = [] #rows list of dict with layer and BC info
        #for each layer add data to dataframe.
        i=0
        i_sol=0
        for layer in self.layers:
            row={}
            row['object']='layer'
            for (key,val) in vars(layer).items():
                row[key]=val
            row['r_inner [m]']=self.radii[i]
            row['r_outer [m]']=self.radii[i+1]
            if layer.layer_type=='solid':
                Ti = self.T(self.radii[i])
                To = self.T(self.radii[i+1]*(1-1e-10))
                row['T_{r_inner} [degC]'] = Ti
                row['T_{r_outer} [degC]'] = To
                row['T_min [degC]'] = min([Ti,To])
                #check if there is the possibility of a max temp higher than
                #boundary temperatures (i.e. if you set dT/dr=0 do you get
                #a real radius value... for this CA must be positive.
                if self.C_[2*i_sol]>0:
                    r_Tmax = (2*layer.k*self.C_[2*i_sol]/layer.V_heat)**0.5
                    if r_Tmax<=self.radii[i+1] and r_Tmax>=self.radii[i]:
                        row['T_max [degC]']=self.T(r_Tmax)
                    else:
                        row['T_max [degC]']=max([Ti,To])
                else:
                    row['T_max [degC]']=max([Ti,To])
                Qi = self.Q(self.radii[i])
                Qo = self.Q(self.radii[i+1]*(1-1e-10))
                Qvol = layer.V_heat*np.pi*(self.radii[i+1]**2-self.radii[i]**2)
                row['net heat transfer [W/m]']=Qi-Qo+Qvol
                row['internal heat generation [W/m]']=Qvol
                row['inner boundary heat transfer [W/m]']=Qi
                row['outer boundary heat transfer [W/m]']=Qo
                i_sol+=1
            elif layer.layer_type=='fluid':
                Ti = self.T(self.radii[i]*(1-1e-10))
                To = self.T(self.radii[i+1])
                row['T_{r_inner} [degC]']=Ti
                row['T_{r_outer} [degC]']=To
                row['T_max [degC]']=max([Ti,To,layer.T_inf])
                row['T_min [degC]']=min([Ti,To,layer.T_inf])
                Qi = self.Q(self.radii[i]*(1-1e-10))
                Qo = self.Q(self.radii[i+1])
                Qnet = Qi-Qo
                row['inner boundary heat transfer [W/m]']=Qi
                row['outer boundary heat transfer [W/m]']=Qo
                row['net heat transfer into layer/boundary [W/m]']=Qnet
            rows.append(pd.Series(data=row))
            i+=1
        for BC in self.BC:
            row={}
            row['object']='boundary condition'
            for (key,val) in vars(BC).items():
                if key=='h' and BC.BC_side=='outer':
                    key='h_in'
                elif key=='h' and BC.BC_side=='inner':
                    key='h_out'
                row[key]=val
            if BC.BC_side == 'inner':
                Qnet = -self.Q(self.radii[0])
            elif BC.BC_side =='outer':
                Qnet = self.Q(self.radii[-1]*(1-1e-10))
            row['net heat transfer into layer/boundary [W/m]']=Qnet
            rows.append(pd.Series(data=row))
        #readable headings
        column_head={'t':'layer thickness [m]',
            'layer_type':'layer type',
            'k':'conductivity [W/mK]',
            'V_heat':'volumetric heat generation [W/m^3]',
            'R_out':'thermal resistance of outer layer interface [m^2K/W]',
            'BC_side':'boundary condition side',
            'BC_type':'boundary condition type',
            'h_in':'inner heat transfer coefficient [W/m^2K]',
            'h_out':'outer heat transfer coefficient [W/m^2K]',
            'T_inf':'Bulk temperature [degC]'
            }
        df=pd.DataFrame(rows)
        df=df.rename(columns=column_head)
        return df
a = solid_layer(0.5e-3,30,1e7,1e-5,'Eurofer wall')
b = fluid_layer(1e-3,480,1e4,1e4, 'He interlayer')
c = solid_layer(0.2e-3,10,2e7,1e-5, 'Surface coating')
BC1 = boundary_condition('temperature', 'outer',T_inf=450, R=1e-6)
BC2 = boundary_condition('fluid', 'inner', T_inf=600, h=1e4)
BC3 = boundary_condition('fluid', 'outer', T_inf=400, h=1e4, label='outer fluid')
BC4 = boundary_condition('fluid', 'inner', T_inf=400, h=1e4, label='inner fluid')
BCHFout = boundary_condition('heat flux', 'outer',q=1e6)
BCHFin = boundary_condition('heat flux', 'inner',q=-1e6)
A = radial_heat_transfer_1d(5e-3,[c,a,c,b,c,a,c],BC3,BC2)
B = radial_heat_transfer_1d(5e-3,[c,a,c],BC3,BCHFout)
C=radial_heat_transfer_1d(5e-3,[c,a,c],BC3,BC4)
c4_1=solid_layer(0.5e-3,300,0,1e-5)
c4_2=solid_layer(3.5e-3,30,1e8,0)
Comp_c4 = radial_heat_transfer_1d(5e-3,[c4_1,c4_2,b,c4_1,c4_2],BC2,BC3)
