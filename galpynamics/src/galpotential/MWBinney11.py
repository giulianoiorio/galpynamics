from __future__ import division, print_function
from ..pot_disc.Exponential_disc import  Exponential_disc
from ..pot_halo.NFW_halo import  NFW_halo
from ..pot_halo.alfabeta_halo import  alfabeta_halo
from .galpotential import galpotential
import numpy as np

class MWBinney11(galpotential):
    
    def __init__(self):
        
        self.modified=False
        self.name='MWBinney11'
        
        #Halo
        d0=1.32e7
        rs=16.47
        mcut=100
        q=1.0
        e=np.sqrt(1-q*q)
        halo=NFW_halo(d0=d0, rs=rs, mcut=mcut, e=e)
        #Thin disc
        sigma0=7.68e8
        Rd=2.64
        zd=0.3
        Rcut=50
        zcut=50
        tnd=Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd, Rcut=Rcut, zcut=zcut,zlaw='exp')
        #Thick disc
        sigma0=2.01e8
        Rd=2.97
        zd=0.9
        Rcut=50
        zcut=50
        tkd=Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd, Rcut=Rcut, zcut=zcut,zlaw='exp')
        #Bulge
        d0=9.49e10
        rs=0.075
        alfa=0
        beta=1.8
        mcut=2.5
        q=0.5
        e=np.sqrt(1-q*q)
        bulge=alfabeta_halo(d0=d0,  alfa=alfa, beta=beta, rs=rs,mcut=mcut,e=e)
        
        super(MWBinney11,self).__init__(dynamic_components=(tkd,tnd,bulge,halo))
        
        
    def remove_components(self,idx=()):

        dynamic_components=[]

        for i in range(len(self.dynamic_components)):

            for j in idx:
                if i!=j:
                    print('i',i,idx)
                    dynamic_components.append(self.dynamic_components[i])
                else:
                    pass

        self.dynamic_components=dynamic_components

        self.ncomp = len(self.dynamic_components)
        
        if len(idx)>0:
            self.modified=True

        return 0

    def add_components(self,components=()):

        self._check_components(components)

        self.dynamic_components=self.dynamic_components+list(components)

        self.ncomp=len(self.dynamic_components)
        
        if len(components>0):
            self.modified=True

        return 0
        
        
    def __str__(self):
        s='%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n'
        if self.modified:   s+='MW Model:  Binney+11 (modified) \n'
        else: s+='MW Model: Binney+11\n'
        s+='Number of dynamical components: %i \n'%self.ncomp
        for i,comp in enumerate(self.dynamic_components):
            s+='-------------------\n'
            s+='Components: %i \n'%i
            s+=comp.__str__()
            s+='-------------------\n'
            i+=1
        s+='%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n'
                
            
        return s
        