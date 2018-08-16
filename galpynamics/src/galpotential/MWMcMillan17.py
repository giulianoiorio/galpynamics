from __future__ import division, print_function
from ..pot_disc.Exponential_disc import  Exponential_disc
from ..pot_disc.McMillan_disc import  McMillan_disc
from ..pot_halo.NFW_halo import  NFW_halo
from ..pot_halo.alfabeta_halo import  alfabeta_halo
from .galpotential import galpotential
import numpy as np

#TODO: Add the HI and H2 disc components
class MWMcMillan17(galpotential):
    
    def __init__(self):
        
        self.modified=False
        self.name='MWMcMillan17'
        
        #Halo
        d0=8.53702e+06
        rs=19.5725
        mcut=500
        q=1.0
        e=np.sqrt(1-q*q)
        halo=NFW_halo(d0=d0, rs=rs, mcut=mcut, e=e)
        #HI disc
        sigma0=53.1e6 #Msun/Kpc^2
        Rd=7
        Rm=4
        zd=0.085*2
        Rcut=50
        zcut=50
        HId=McMillan_disc.thick(sigma0=sigma0, Rd=Rd, Rm=Rm, zd=zd, Rcut=Rcut, zcut=zcut, zlaw='sech2')
        #H2 disc
        sigma0=2180e6 #Msun/Kpc^2
        Rd=1.5
        Rm=12
        zd=0.045*2
        Rcut=50
        zcut=50
        H2d=McMillan_disc.thick(sigma0=sigma0, Rd=Rd, Rm=Rm, zd=zd, Rcut=Rcut, zcut=zcut, zlaw='sech2')
        #Thin disc
        sigma0=8.95679e+08
        Rd=2.49955
        zd=0.3
        Rcut=50
        zcut=50
        tnd=Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd, Rcut=Rcut, zcut=zcut,zlaw='exp')
        #Thick disc
        sigma0=1.83444e+08
        Rd=3.02134
        zd=0.9
        Rcut=50
        zcut=50
        tkd=Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd, Rcut=Rcut, zcut=zcut,zlaw='exp')
        #Bulge
        d0=9.8351e+10
        rs=0.075
        alfa=0
        beta=1.8
        mcut=2.1
        q=0.5
        e=np.sqrt(1-q*q)
        bulge=alfabeta_halo(d0=d0,  alfa=alfa, beta=beta, rs=rs,mcut=mcut,e=e)
        
        super(MWMcMillan17,self).__init__(dynamic_components=(tkd,tnd,HId, H2d, bulge,halo))
        
        
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
        if self.modified:   s+='MW Model:  McMillan+17 (modified) \n'
        else: s+='MW Model: McMillan+17\n'
        s+='Number of dynamical components: %i \n'%self.ncomp
        for i,comp in enumerate(self.dynamic_components):
            s+='-------------------\n'
            s+='Components: %i \n'%i
            s+=comp.__str__()
            s+='-------------------\n'
            i+=1
        s+='%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n'
                
            
        return s
        