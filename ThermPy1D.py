# -*- coding: utf-8 -*-
"""
Attempt to create a generalised 1-D code for heat transfer for linear and
cylindrical systems.

The Thermal1D class is used to assemble and solve the system of equations - 
it is generally not for use by the user who should instead use the child
Linear1D and Radial1D classes.

The Linear1D and Radial1D classes contain the specific continuity equations
and boundary conditions to be assembled and solved Thermal1D. There is inbuilt
plotting and summary dataframe provided.

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
heat transfer at any position can be trivially calculated using the inbuilt
methods.
@author: bruce
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
class layer:
    """
    Parent class for the other layer types, takes care of some methods like 
    assigning common variables.
    
    Parameters
    ----------
        t: float
            Required; the layer thickness [m]
        label: str
            Optional; a descriptive string used for labeling the layer.
    """
    def __init__(self,  t, label):
        if t<=0:
            raise ValueError('please enter positive non-zero layer thickness for {}'.format(self))
        self.t = t #layer thickness [m]
        self.label=label
class solid_layer(layer):
    """
    Class to manage input assignment for solid layers such as metals and
    ceramics.
    
    Parameters
    ----------
        t: float
            Required; the layer thickness [m]
        k: float
            Required; thermal conductivity [W/mK]
        V_heat: float
            Optional; volumetric heat generation [W/m^3], default 0.
        R_out: float
            Optional; thermal resistance of the outer interface [m^2K/W]
        label: str
            Optional; a descriptive string used for labeling the layer.
    """
    def __init__(self, t, k, V_heat = 0, R_out = 0, label=None):
        super().__init__(t,label)
        self.layer_type = 'solid'
        self.k = k #conductivity in solid [W/kg/K]
        self.V_heat = V_heat #volumetric heating [W/m^3] 
        self.R_out = R_out #thermal resistance at outer edge [m^2K/W]
class fluid_layer(layer):
    """
    Class to manage variable assignment of fluid layers. The assumption for
    fluid layers is that the coolant flow rate is sufficient to prevent
    significant axial thermal gradients.
    
    Parameters
    ----------
        t: float
            Required; the layer thickness [m]
        T_inf: float
            Required; the bulk coolant temperature [K]
        h_in: float
            Required; heat transfer coefficient on inner annulus [W/m^2K]
        h_out: float
            Required; heat transfer coefficient on outer annulus [W/m^2K]
        label: str
            Optional; a descriptive string used for labeling the layer.
    """
    def __init__(self, t, T_inf, h_in, h_out, label=None):
        super().__init__(t, label)
        self.layer_type = 'fluid'
        self.T_inf = T_inf #bulk coolant temperature [K]
        if (h_in <= 0 or h_out <= 0):
            raise ValueError('Please enter positive, non-zero HTC values for {}'.format(self))
        self.h_in = h_in #HTC at inner radial surface [W/m^2K]
        self.h_out = h_out #HTC at outer radial surface [W/m^2K]
class boundary_condition:
    """
    Class for boundary conditions, takes care of some methods like assigning
    common variables.
    
    Parameters
    ----------
    Required keyword arguments depend on the boundary condition type supplied.
        BC_type: str ('temperature', 'fluid' or 'heat flux'')
            Required; defines the type of boundary condition.
        BC_side: str ('inner' or 'outer')
            Required; applies the boundary condition to the inner or outer
            edge of the assembly.
        **kwargs:
            For BC_type 'temperature':
                T_inf; float
                    Required: the temperature of the boundary [K].
                R; float
                    Optional: the contact resistance [m^2K/W], default 0.
                label; str
                    Optional: a descriptive string used for labeling the layer.
            For BC_type 'fluid':
                T_inf; float
                    Required: the bulk coolant temperature [K].
                h; float
                    Required: Heat transfer coefficient [W/m^2K] cannot be 0.
                label; str
                    Optional: a descriptive string used for labeling the layer.
            For BC_type 'heat flux':
                q; float
                    Required: heat flux applied to the boundary in radial dir,
                    [W/m^2]
                label; str
                    Optional: a descriptive string used for labeling the layer.
                
    """
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
            elif val == 0 and (key not in kwargs):
                if key == 'label':
                    self.label = None
                elif key == 'R':
                    setattr(self, key, 0)
        # if not hasattr(self,'label'):
        #     self.label=None
      
class Thermal1D:
    """
    Generic 1D class with common methods useful to 1D thermal systems.
    Includes the building and solving of the simultaneous equations to find
    the integration constants required for each solid layer.
    """            
    def __init__(self, layers, BC_1, BC_2):
        if layers[0].layer_type == 'fluid':
            raise RuntimeError('First layer cannot be a "fluid_layer" object.\n'+
                              'Use fluid inner boundary condition instead.')
        if layers[-1].layer_type == 'fluid':
            raise RuntimeError('Final layer cannot be a "fluid_layer" object.\n'+
                              'Use fluid outer boundary condition instead.')
        if (BC_1.BC_side==BC_2.BC_side) and ((BC_1.BC_type!='heat flux') and (BC_2.BC_type!='heat flux')):
            raise RuntimeError('Cannot have two BCs at same boundary unless 1 is type "heat flux".')
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
        self.solid_map = solid_map
        self.fluid_map = fluid_map
        if (BC_1.BC_type == 'heat flux') and (BC_2.BC_type == 'heat flux') and 1 not in self.fluid_map:
            raise RuntimeError('Cannot have two heat flux boundary conditions.')
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
        Assigns integration constants to fully define temperature profile,
        two are needed to define the temperature profile in each solid layer.
        The system of equations are set up using 2 boundary conditions,
        A heat continuity equation for each solid-solid boundary.
        A temperature equation for each boundary.
        The form of the equation solved is AC=B, where C (the array of
        integration constants are to be solved for). The least squares method
        is used as with certain layer combinations result in singular A
        matrices.
        -------
        """
        #initialise matrices
        N=self.solid_map.count(1) #count number of solid layers
        A_, B_ = np.zeros((2*N,2*N)), np.zeros((2*N,1))
        j=0 # row tracker
        #apply boundary conditions:
        for BC_ in self.BC:
            if BC_.BC_side == 'inner':
                n, sgn=0, 1 # first layer index, sign for heat resistance term
                r = self.distance[0]
                V = self.layers[0].V_heat
                k = self.layers[0].k
            else:
                n, sgn=N-1,-1 #last layer index, sign for heat resistance term
                r = self.distance[-1]
                V = self.layers[-1].V_heat
                k = self.layers[-1].k
            if BC_.BC_type!='heat flux':
                if BC_.BC_type=='temperature':
                    R= BC_.R
                else:
                    R= 1/BC_.h
                A_[j,2*n:2*n+2], B_[j,0] = self.temperature_bound(
                        sgn,r,k,R,V,BC_.T_inf)
                j+=1
            else: #for heat flux BC
                A_[j,2*n:2*n+2], B_[j,0] = self.heatflux_bound(r,k,BC_.q,V)
                j+=1    
        #apply layer temp and heat continuity
        for i in range(len(self.layers)-1):
            i_sol = sum(self.solid_map[0:i]) #solid layer index
            if self.layers[i].layer_type == self.layers[i+1].layer_type:
                #statement only triggered for solid-solid interfaces as fluid-fluid layer arrangements are not allowed.
                #heat continuity
                r=self.distance[i+1]
                k0=self.layers[i].k
                k1=self.layers[i+1].k
                V0=self.layers[i].V_heat
                V1=self.layers[i+1].V_heat
                R=self.layers[i].R_out
                A_[j,2*i_sol:2*i_sol+4], B_[j,0] = self.heat_continuity(
                        r,k0,k1,V0,V1)
                j+=1
                #temp continuity
                A_[j,2*i_sol:2*i_sol+4], B_[j,0] = self.temp_continuity(
                        r,k0,k1,V0,V1,R)              
                j+=1
            elif self.layers[i].layer_type == 'solid':
                #only triggered when going from solid-fluid
                #define based on solid-fluid transition
                r=self.distance[i+1] 
                k=self.layers[i].k
                V=self.layers[i].V_heat
                R=1/(self.layers[i+1].h_in)
                T_inf = self.layers[i+1].T_inf
                A_[j,2*i_sol:2*i_sol+2],B_[j,0]=self.temperature_bound(
                        -1,r,k,R,V,T_inf)
                j+=1
                #second boundary condition (fluid-solid)
                r=self.distance[i+2]
                k=self.layers[i+2].k
                V=self.layers[i+2].V_heat
                R=1/(self.layers[i+1].h_out)
                T_inf = self.layers[i+1].T_inf
                A_[j,2*(i_sol+1):2*(i_sol+1)+2],B_[j,0] = self.temperature_bound(
                        1,r,k,R,V,T_inf)
                j+=1
            else:
                pass
        self.A_=A_ #store LHS
        self.B_=B_ #store RHS
        C_ = np.linalg.lstsq(A_,B_,rcond=None)[0] #solve integration constants
        self.C_=C_ #store result as attribute
    
    def T(self,distance):
        """      
        Parameters
        ----------
        distance : float
            radial/linear position from center/first edge [m].

        Returns
        -------
        T: float
            Temperature at position [K].
        """
        layer_i = self.layer_index_for_distance(distance)
        if layer_i == 'inner':
            if not self.BC_in:
                return self.T_solid(self.distance[0],self.layers[0],0)[0]
            else:
                for BC_ in self.BC_in:
                    if BC_.BC_type!='heat flux':
                        return BC_.T_inf
                    else:
                        return self.T_solid(self.distance[0],self.layers[0],0)[0]
            #return something if inner temp or fluid BC exists do solid with distance[0] if not
        elif layer_i == 'outer':
            if not self.BC_out:
                return self.T_solid(self.distance[-1],self.layers[-1],len(self.layers)-1)[0]
            else:
                for BC_ in self.BC_out:
                    if BC_.BC_type!='heat flux':
                        return BC_.T_inf
                    else:
                        return self.T_solid(self.distance[-1],self.layers[-1],len(self.layers)-1)[0]
        else: 
            layer = self.layers[layer_i]
            if layer.layer_type == 'fluid':
                return layer.T_inf
        
            elif layer.layer_type=='solid':
                return self.T_solid(distance,layer,layer_i)[0]
    def layer_index_for_distance(self,rad):
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
        if rad < self.distance[0]:
            return 'inner'
        elif rad >= self.distance[-1]:
            return 'outer'
        else: 
            for i in range(len(self.layers)):
                if rad >= self.distance[i] and rad< self.distance[i+1]:
                    return i                
    def plot_T(self, dpoints=100,
               plotQ=1,
               plotq=1,
               show_bounds=1,
               Q_labels=0,
               patches=1,
               set_alpha=0.5
               ):
        """
        Plots the temperature profile through the wall thickness.
        Parameters
        ----------
        dpoints : int
            number of datapoints to plot through whole wall thickness.
            The default is 100.
        plotQ : boolean (0,1, True, False) optional
            If True plots heat transfer through solid layers, else does not.
            The default is 1.
        plotq : boolean (0,1, True, False) optional
            If True plots heat flux through solid layers, else does not.
            The default is 1.
        show_bounds : boolean (0,1, True, False) optional
            If true labels layers. Draw layer boundaries.
            The default is 0.
        Q_labels : boolean (0,1, True, False) optional
            If true labels the internal heat generation for each solid layer
            and the net heat transfer for each fluid layer. 
            The default is 0.
        patches : boolean (0,1, True, False) optional
            If true labels layers by drawing patches in layer and adding a
            legend.
        Returns
        -------
        figure

        """
        T=[]
        axs=[]
        if (self.distance[0] == 0) and (self.analysis_type == 'radial'):
            distances = np.linspace(0 + self.distance[1]/1e9,
                                    self.distance[-1]*1.05,
                                    dpoints)
        else:
            distances = np.linspace(self.distance[0]*0.95,
                                    self.distance[-1]*1.05,
                                    dpoints)
        plotnum=1
        if plotQ==True:
            plotnum+=1
            Q=[]
        if plotq==True:
            plotnum+=1
            q=[]
        for r in distances:
            T.append(self.T(r))
            if plotQ==True:
                Q.append(self.Q(r))
            if plotq==True:
                q.append(self.q(r))
        fig=plt.figure()
        ax1 = fig.add_subplot(111)
        axs.append(ax1)
        if plotQ==True:
            Qax = ax1.twinx()
            axs.append(Qax)
            Qax.set_ylabel(r'Heat transfer W/m',color='b')
            Qax.plot(distances*1e3,Q,'b-', label='Heat Transfer')
            Qax.ticklabel_format(style='sci',axis='y', scilimits=(-2,2))
        if plotq==True:
            qax = ax1.twinx()
            axs.append(qax)
            qax.set_ylabel(r'Heat flux W/m$^2$',color='m')
            qax.plot(distances*1e3,q,'m-', label='Heat Flux')
            qax.ticklabel_format(style='sci',axis='y', scilimits=(0,0))
        if show_bounds==1:
            ax1.plot([self.distance[0]*1e3,self.distance[0]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            ax1.plot([self.distance[-1]*1e3,self.distance[-1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
            for i in range(len(self.layers)-1):
                if self.layers[i].layer_type==self.layers[i+1].layer_type:
                    ax1.plot([self.distance[i+1]*1e3,self.distance[i+1]*1e3],[min(T),max(T)],'k-.',linewidth=0.5)
                else:
                    ax1.plot([self.distance[i+1]*1e3,self.distance[i+1]*1e3],[min(T),max(T)],'k-',linewidth=0.5)
        if self.analysis_type in ['radial', 'spherical']:
            ax1.set_xlabel('Radius [mm]')
        elif self.analysis_type == 'linear':
            ax1.set_xlabel('Thickness [mm]')
        ax1.set_ylabel(r'Temperature [K]', color='r')
        ax1.plot(distances*1e3,T,'r-', label='Temperature')
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
            ax1.annotate(r'$Q_{i,net}$='+'{:.3e}'.format(-self.Q(self.distance[0]))
                         +' W/m',
                         xy=(0.05,0.5),
                         rotation='vertical',
                         xycoords='axes fraction',
                         verticalalignment='center',
                         horizontalalignment='center')
            ax1.annotate(r'$Q_{o,net}$='+'{:.3e}'.format(self.Q(self.distance[-1]*(1-1e-10)))
                         +' W/m',
                         xy=(0.95,0.5),
                         rotation='vertical',
                         xycoords='axes fraction',
                         verticalalignment='center',
                         horizontalalignment='center')
            i=0
            for layer in self.layers:
                if layer.layer_type=='solid':
                    if self.analysis_type == 'radial':
                        Qvol = np.pi*(self.distance[i+1]**2-self.distance[i]**2)*layer.V_heat
                    elif self.analysis_type == 'linear':
                        Qvol = self.width*layer.t*layer.V_heat
                    ax1.annotate(r'$Q_{nuc}$='+'{:.3e}'.format(Qvol)
                         +' W/m',
                         xy=((sum(self.distance[i:i+2])*1e3/2),
                             ycenter),
                         rotation='vertical',
                         xycoords='data',
                         verticalalignment='center',
                         horizontalalignment='center',
                         )
                elif layer.layer_type=='fluid':
                    Qnet = self.Q(self.distance[i]*(1-1e-10))-self.Q(self.distance[i+1])
                    ax1.annotate(r'$Q_{net,fluid}$='+'{:.3e}'.format(Qnet)
                         +' W/m',
                         xy=((sum(self.distance[i:i+2])*1e3/2),
                             ycenter),
                         rotation='vertical',
                         xycoords='data',
                         verticalalignment='center',
                         horizontalalignment='center',
                         )
                i+=1
        #Make some patches with labels and a legend.
        if patches==1:
            #find unique layers
            unique_layers=set(self.layers)
            unique_labels={} #init dicts for storing unique labels
            i={} #count number of identical labels for different objects
            i['unnamed layer'],i['unnamed BC']=1,1 #add counter for unlabelled layers
            #Assign labels to each unique layer.
            for layer in unique_layers:
                try:
                    if layer.label==None:
                        unique_labels[layer]='unnamed layer {}'.format(i['unnamed layer'])
                        i['unnamed layer']+=1
                    else:
                        unique_labels[layer]=(layer.label+' {}'.format(i[layer.label]))
                        i[layer.label]+=1
                except KeyError:
                    unique_labels[layer]=layer.label
                    i[layer.label]=2
            #Assign labels to each unique BC.
            for BC in self.BC:
                if BC.label==None:
                    unique_labels[BC]='unnamed BC {}'.format(i['unnamed BC'])
                    i['unnamed BC']+=1
                else:
                    try:
                        unique_labels[BC]=(BC.label+' BC {}'.format(i[BC.label+' BC']))
                        i[BC.label+' BC']+=1
                    except KeyError:
                        unique_labels[BC]=BC.label + ' BC'
                        i[BC.label+' BC']=2
            N=len(unique_labels)
            patch_list=[]
            i=0
            col = {} #colours corresponding to labels
            prop = np.linspace(0,1,N) #equispaced sampling for colour spetrum
            for (key,val) in sorted(unique_labels.items(), key=lambda x: x[1]):
               col[key]=cm.rainbow(prop[i],alpha=set_alpha) 
               i+=1
            i=0
            ymin = ax1.get_ylim()[0]
            yrange = ax1.get_ylim()[1]-ymin
            for layer in self.layers:
                colour = col[layer]
                ax1.add_patch(Rectangle((self.distance[i]*1e3,ymin),layer.t*1e3,
                              yrange,
                              color=colour,
                              label=unique_labels[layer]))
                i += 1
            if len(self.BC_in)==1:
                colour=col[self.BC_in[0]]
                xmin = ax1.get_xlim()[0]
                xmax = ax1.get_xlim()[1]
                xrange_inner = self.distance[0]*1e3-xmin
                xrange_outer = xmax-self.distance[-1]*1e3
                ax1.add_patch(Rectangle((xmin,ymin),xrange_inner,
                              yrange,
                              color=colour,
                              label=unique_labels[self.BC_out[0]]))
                colour=col[self.BC_out[0]]
                ax1.add_patch(Rectangle((self.distance[-1]*1e3,ymin),xrange_outer,
                              yrange,
                              color=colour,
                              label=unique_labels[self.BC_out[0]]))
                # patch_list.append(Rectangle((0,0),0,0,
                #                             color=col[self.BC_in[0]],
                #                             label=unique_labels[self.BC_in[0]]))
                # patch_list.append(Rectangle((0,0),0,0,
                #                             color=col[self.BC_out[0]],
                #                             label=unique_labels[self.BC_out[0]]))
            for (layer, label) in sorted(unique_labels.items(), key=lambda x: x[1]):
                patch_list.append(Rectangle((0,0),0,0,
                                            color=col[layer],
                                            label=unique_labels[layer]))

        ax1.legend(handles=patch_list, loc='best')
        fig.tight_layout()
        return fig, axs

    def results(self):
        """
        Returns a dataframe object with summarising all the inputs and main
        temperature results for easy export to a variety of formats.
        """
        rows = [] #rows list of dict with layer and BC info
        #for each layer add data to dataframe.
        i=0
        i_sol=0
        if self.analysis_type in ['radial', 'spherical']:
            coord='r'
        elif self.analysis_type == 'linear':
            coord='x'
        if (self.distance[0] == 0) and self.analysis_type in ['radial', 'spherical']: #to supress divide by zero errors
            self.distance[0] = 0 + self.distance[1]/1e9
            self.zero_radial = True
        else:
            self.zero_radial = False
        for layer in self.layers:
            row={}
            row['object']='layer'
            for (key,val) in vars(layer).items():
                row[key]=val
            row['{}_inner [m]'.format(coord)]=self.distance[i]
            row['{}_outer [m]'.format(coord)]=self.distance[i+1]
            if layer.layer_type=='solid':
                Ti = self.T(self.distance[i])
                To = self.T(self.distance[i+1]*(1-1e-10))
                row['T_({}_inner) [K]'.format(coord)] = Ti
                row['T_({}_outer) [K]'.format(coord)] = To
                row['T_min [K]'] = min([Ti,To])
                #check if there is the possibility of a max temp higher than
                #boundary temperatures (i.e. if you set dT/dr=0 do you get
                #a real radius value... for this CA must be positive.
                if self.C_[2*i_sol]>0:
                    if self.analysis_type == 'radial':
                        r_Tmax = (2*layer.k*self.C_[2*i_sol]/layer.V_heat)**0.5
                    elif self.analysis_type == 'linear':
                        r_Tmax = layer.k*self.C_[2*i_sol]/layer.V_heat
                    if r_Tmax<=self.distance[i+1] and r_Tmax>=self.distance[i]:
                        row['T_max [K]']=self.T(r_Tmax)
                    else:
                        row['T_max [K]']=max([Ti,To])
                else:
                    row['T_max [K]']=max([Ti,To])
                Qi = self.Q(self.distance[i])
                Qo = self.Q(self.distance[i+1]*(1-1e-10))
                if self.analysis_type == 'radial':
                    Qvol = layer.V_heat*np.pi*(self.distance[i+1]**2-self.distance[i]**2)
                elif self.analysis_type == 'linear':
                    Qvol = layer.V_heat*self.width*layer.t
                row['net heat transfer [W/m]']=Qi-Qo+Qvol
                row['internal heat generation [W/m]']=Qvol
                row['inner boundary heat transfer [W/m]']=Qi
                row['outer boundary heat transfer [W/m]']=Qo
                i_sol+=1
            elif layer.layer_type=='fluid':
                Ti = self.T(self.distance[i]*(1-1e-10))
                To = self.T(self.distance[i+1])
                row['T_({}_inner) [K]'.format(coord)]=Ti
                row['T_({}_outer) [K]'.format(coord)]=To
                row['T_max [K]']=max([Ti,To,layer.T_inf])
                row['T_min [K]']=min([Ti,To,layer.T_inf])
                Qi = self.Q(self.distance[i]*(1-1e-10))
                Qo = self.Q(self.distance[i+1])
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
                Qnet = -self.Q(self.distance[0])
            elif BC.BC_side =='outer':
                Qnet = self.Q(self.distance[-1]*(1-1e-10))
            row['net heat transfer into layer/boundary [W/m]']=Qnet
            rows.append(pd.Series(data=row))
        if self.zero_radial == True:
                self.distance[0] = 0
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
            'T_inf':'Bulk temperature [K]',
            'q':'boundary heat flux [W/m^2]',
            'R':'boundary thermal resistance [m^2K/W]'
            }
        df=pd.DataFrame(rows) #make the dataframe
        df=df.rename(columns=column_head) #rename the column headings
        return df
            
class Linear1D(Thermal1D):
    """
    Class for building up system of equations, solving the integration
    constants for the solid layers.
    
    Parameters
    ----------
        layers: list of layer objects
            The layers from inner to outer radius inside a list.
        BC_1: boundary_layer object
            The first boundary layer.
        BC_1: boundary_layer object
            The second boundary layer. 
        width: float
            optional. Default is 1.0.
    Returns
    ----------
    Linear1D object.
    
    Methods to evaluate the temperature, heat flux and heat transfer at any
    radial position.
    Method to plot the temperature, heat flux and heat transfer data.
    Method to return a dataframe of inputs and most results of interest to be
    eacily manipulated, filtered and exported in desired data format using
    pandas inbuilt methods.
    """                
    def __init__(self, layers, BC_1, BC_2, width = 1.0):
        super().__init__(layers, BC_1, BC_2)
        distance = [0]
        for i in range(len(layers)):
            distance.append(distance[i]+layers[i].t)
        self.distance = distance
        self.analysis_type = 'linear'
        self.width=width
    def temp_continuity(self,x,k0,k1,V0,V1,R):
        """
        Returns LHS and RHS of temperature continuity equation at each solid-
        solid layer interface.

        Parameters
        ----------
        x : float
            Layer interface x-coord [m] (first layer first boundary at x=0)
        k0 : float
            thermal conductivity of ith solid layer [W/mK]
        k1 : float
            thermal conductivity of ith+1 solid layer [W/mK]
        V0 : float
            Volumetric heat generation of ith solid layer [W/m^3]
        V1 : flaot
            Volumetric heat generation of ith+1 solid layer [W/m^3]
        R : float
            Thermal contact resistance of layer interface [m^2.K/W]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        A = np.array([[x+k0*R, 1, -x, -1]])
        B = x*(V0*(R+x/(2*k0))-V1*x/(2*k1))
        return A, B
    def heat_continuity(self,x,k0,k1,V0,V1):
        """
        Returns LHS and RHS of heat continuity equation at each solid-
        solid layer interface.

        Parameters
        ----------
        x : float
            Layer interface x-coord [m] (first layer first boundary at x=0)
        k0 : float
            thermal conductivity of ith solid layer [W/mK]
        k1 : float
            thermal conductivity of ith+1 solid layer [W/mK]
        V0 : float
            Volumetric heat generation of ith solid layer [W/m^3]
        V1 : flaot
            Volumetric heat generation of ith+1 solid layer [W/m^3]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        A = np.array([[-k0,0,k1,0]])
        B = x*(V1-V0)
        return A, B
    def temperature_bound(self,sgn,x,k,R,V,T_inf):
        """
        Returns LHS and RHS of temperatyre boundary condition. Can be used
        to represent a temperature boundary condition, a thermal contact
        resistance between a solid-solid interface or a convection boundary
        at a fluid-solid or solid-fluid interface.

        Parameters
        ----------
        sgn : 1, -1
            1 for boundary-solid interfaces, -1 for solid-boundary interfaces
        x : float
            Layer interface x-coord [m] (first layer first boundary at x=0)
        k : float
            thermal conductivity of solid layer [W/mK]
        R : float
            Thermal resistance (1/heat transfer coefficient) [m^2 K / W]
        V : flaot
            Volumetric heat generation in solid layer [W/m^3]
        T_inf : float
            Bulk temperature of solid or fluid boundary (wall temperature
            when R=0) [K]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        A = np.array([[sgn*k*R-x, -1]])
        B = V*x*(sgn*R-x/(2*k)) - T_inf
        return A, B
    def heatflux_bound(self,x,k,q,V):
        """
        LHS and RHS of heat flux boundary condition.
        
        Parameters
        ----------
        x : float
            Layer interface x-coord [m] (first layer first boundary at x=0)
        k : float
            thermal conductivity of solid layer [W/mK]
        q : float
            heat flux applied to surface.
        V : flaot
            Volumetric heat generation in solid layer [W/m^3]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.
        """
        A = np.array([[1, 0]])
        B = -1/k*(q - V*x)
        return A, B
    def T_solid(self,x,layer,layer_i):
        """
        Parameters
        ----------
        x : float
            distance from first layer outer face [m].
        layer : solid_layer object
            The solid layer in which to calculate the temperature.
        layer_i : int
            layer index of the layer in which to calculate the temperature.

        Returns
        -------
        float
            Temperature [K]
        """
    
        if not hasattr(self,'C_'):
            self.C()
        V_heat, k = layer.V_heat, layer.k
        solid_layer_i = sum(self.solid_map[0:layer_i])
        CA, CB = self.C_[2*solid_layer_i],self.C_[2*solid_layer_i+1]
        return -V_heat*x**2/(2*k) + CA*x + CB 
    def q(self,distance):
        """
        Returns heat flux [W/m^2] given any position [m].
        
        Parameters
        ----------
        distance : float
            distance from first layer outer face [m].

        Returns
        -------
        q: float, np.nan
            heat flux [W/m^2]. Returns np.nan if not in defined system range.
        """
        #calculating heat flux at a radial position
        if not hasattr(self,'C_'):
            self.C()
        layer_i = self.layer_index_for_distance(distance)
        if layer_i == 'outer' or layer_i == 'inner':
            return np.nan
        elif self.layers[layer_i].layer_type == 'fluid':
            return np.nan
        else:
            layer = self.layers[layer_i]
            V_heat, k = layer.V_heat, layer.k
            solid_layer_i = sum(self.solid_map[0:layer_i])
            CA = self.C_[2*solid_layer_i][0]
            return self.q_solid(distance, V_heat, k, CA)
    def q_solid(self,distance, V_heat, k, CA):
        """
        Parameters
        ----------
        distance : float
            distance from first layer outer face [m].
        V_heat : flaot
            Volumetric heat generation in solid layer [W/m^3]
        k : flaot
            thermal conductivity of solid layer [W/mK]
        CA : float
            first integation constant [K/m]

        Returns
        -------
        float
            Heat flux in a solid [W/m^2] at given position [m]..

        """
        return -k*(-V_heat*distance/k + CA)
    def Q(self, distance):
        """
        Returns heat transfer for unit length [W/m] given any radial position
        [m]
        
        Parameters
        ----------
        distance : float
            x-coord from first layer [m].

        Returns
        -------
        q: float
            heat transfer [W/m]. Returns np.nan if not in defined system range.
        """
        #calculating heat transfer at a radial position
        return self.q(distance)*self.width
    
class Radial1D(Thermal1D):
    """
    Class for building up system of equations, solving the integration
    constants for the solid layers.
    
    Parameters
    ----------
        rad_inner: float
            the inner radius of the system in meters.
        layers: list of layer objects
            The layers from inner to outer radius inside a list.
        BC_1: boundary_layer object
            The first boundary layer.
        BC_1: boundary_layer object
            The second boundary layer.    
    Returns
    ----------
    Radial1D object.
    
    Methods to evaluate the temperature, heat flux and heat transfer at any
    radial position.
    Method to plot the temperature, heat flux and heat transfer data.
    Method to return a dataframe of inputs and most results of interest to be
    eacily manipulated, filtered and exported in desired data format using
    pandas inbuilt methods.
    """
    def __init__(self, rad_inner, layers, BC_1, BC_2):
        super().__init__(layers, BC_1, BC_2)
        if rad_inner<0:
            raise ValueError('rad_inner must be positive.')
        elif rad_inner == 0:
            if (((self.BC[0].BC_side == 'inner') and 
                 (self.BC[0].BC_type == 'heat flux')) or 
                ((self.BC[1].BC_side == 'inner') and 
                 (self.BC[1].BC_type == 'heat flux'))):
                #rad_inner = 0 + self.layers[0].t/1e6 #for maths to work
                for BC in self.BC:
                    if ((BC.BC_side == 'inner') and 
                        (BC.BC_type == 'heat flux') and
                        (BC.q!=0)):
                        print('for inner radius == 0, consider setting q=0 '+ 
                              'for inner BC for physical result')
            else:
                raise ValueError('must apply inner heat flux BC for inner rad == 0')
        distance = [rad_inner]
        for i in range(len(layers)):
            distance.append(distance[i]+layers[i].t)
        self.distance = distance
        self.radii = distance
        self.analysis_type = 'radial'
    def temp_continuity(self,r,k0,k1,V0,V1,R):
        """
        Returns LHS and RHS of temperature continuity equation at each solid-
        solid layer interface.

        Parameters
        ----------
        r : float
            Layer interface radius [m]
        k0 : float
            thermal conductivity of ith solid layer [W/mK]
        k1 : float
            thermal conductivity of ith+1 solid layer [W/mK]
        V0 : float
            Volumetric heat generation of ith solid layer [W/m^3]
        V1 : flaot
            Volumetric heat generation of ith+1 solid layer [W/m^3]
        R : float
            Thermal contact resistance of layer interface [m^2.K/W]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        A = np.array([[np.log(r)+R*k0/r, 1, -np.log(r), -1]])
        B = r/2*(V0*(R+r/(2*k0))-V1*r/(2*k1))
        return A, B
    def heat_continuity(self,r,k0,k1,V0,V1):
        """
        Returns LHS and RHS of heat continuity equation at each solid-
        solid layer interface.

        Parameters
        ----------
        r : float
            Layer interface radius [m]
        k0 : float
            thermal conductivity of ith solid layer [W/mK]
        k1 : float
            thermal conductivity of ith+1 solid layer [W/mK]
        V0 : float
            Volumetric heat generation of ith solid layer [W/m^3]
        V1 : flaot
            Volumetric heat generation of ith+1 solid layer [W/m^3]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        A = np.array([[-k0,0,k1,0]])
        B = r**2/2*(V1-V0)
        return A, B
    def heatflux_bound(self,r,k,q,V):
        """
        LHS and RHS of heat flux boundary condition.
        
        Parameters
        ----------
        r : float
            Layer interface radius [m]
        k : float
            thermal conductivity of solid layer [W/mK]
        q : float
            heat flux applied to surface.
        V : flaot
            Volumetric heat generation in solid layer [W/m^3]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.
        """
        A = np.array([[1, 0]])
        B = -r/k*(q - V*r/2)
        return A, B
    def temperature_bound(self,sgn,r,k,R,V,T_inf):
        """
        Returns LHS and RHS of temperatyre boundary condition. Can be used
        to represent a temperature boundary condition, a thermal contact
        resistance between a solid-solid interface or a convection boundary
        at a fluid-solid or solid-fluid interface.

        Parameters
        ----------
        sgn : 1, -1
            1 for boundary-solid interfaces, -1 for solid-boundary interfaces
        r : ffloat
            Layer interface radius [m]
        k : float
            thermal conductivity of solid layer [W/mK]
        R : float
            Thermal resistance (1/heat transfer coefficient) [m^2 K / W]
        V : flaot
            Volumetric heat generation in solid layer [W/m^3]
        T_inf : float
            Bulk temperature of solid or fluid boundary (wall temperature
            when R=0) [K]

        Returns
        -------
        A : numpy array
            Integration constant coefficients.
        B : float
            RHS of continuity equation.

        """
        if r == 0:
        #avoid divide by zero in zero radius problems
            r = 0 + self.distance[1]/1e9
        A = np.array([[sgn*k*R/r-np.log(r), -1]])
        B = V*r/2*(sgn*R-r/(2*k)) - T_inf
        return A, B
    def q(self,rad):
        """
        Returns heat flux [W/m^2] given any radial position [m].
        
        Parameters
        ----------
        rad : float
            radial position from center [m].

        Returns
        -------
        q: float, np.nan
            heat flux [W/m^2]. Returns np.nan if not in defined system range.
        """
        #calculating heat flux at a radial position
        if not hasattr(self,'C_'):
            self.C()
        layer_i = self.layer_index_for_distance(rad)
        if layer_i == 'outer' or layer_i == 'inner':
            return np.nan
        elif self.layers[layer_i].layer_type == 'fluid':
            return np.nan
        else:
            layer = self.layers[layer_i]
            V_heat, k = layer.V_heat, layer.k
            solid_layer_i = sum(self.solid_map[0:layer_i])
            CA = self.C_[2*solid_layer_i][0]
            return self.q_solid(rad, V_heat, k, CA)
        
    def q_solid(self,rad, V_heat, k, CA):
        """
        Parameters
        ----------
        rad : float
            radial position from center [m].
        V_heat : flaot
            Volumetric heat generation in solid layer [W/m^3]
        k : flaot
            thermal conductivity of solid layer [W/mK]
        CA : float
            first integation constant [K/m]

        Returns
        -------
        float
            Heat flux in a solid [W/m^2] at given position [m]..

        """
        return -k*(-V_heat*rad/(2*k) + CA/rad)
    
    def Q(self,rad):
        """
        Returns heat transfer for unit length [W/m] given any radial position
        [m]
        
        Parameters
        ----------
        rad : float
            radial position from center [m].

        Returns
        -------
        q: float
            heat transfer [W/m]. Returns np.nan if not in defined system range.
        """
        #calculating heat transfer at a radial position
        return self.q(rad)*2*np.pi*rad
        
        return
    def T_solid(self,rad,layer,layer_i):
        """
        Parameters
        ----------
        rad : float
            radial position from center [m].
        layer : solid_layer object
            The solid layer in which to calculate the temperature.
        layer_i : int
            layer index of the layer in which to calculate the temperature.

        Returns
        -------
        float
            Temperature [K]
        """
    
        if not hasattr(self,'C_'):
            self.C()
        V_heat, k = layer.V_heat, layer.k
        solid_layer_i = sum(self.solid_map[0:layer_i])
        CA, CB = self.C_[2*solid_layer_i],self.C_[2*solid_layer_i+1]
        return -V_heat/(4*k)*rad**2+CA*np.log(rad)+CB
    
