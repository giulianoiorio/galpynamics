from __future__ import division, print_function
from .pot_c_ext.core_nfw_halo import potential_core_nfw, vcirc_core_nfw
import multiprocessing as mp
from ..pardo.Pardo import ParDo
from ..utility import cartesian
import numpy as np
import sys
from .pot_halo import halo


class core_NFW_halo(halo):

    def __init__(self,d0,rs,rc,n,e=0,mcut=100):
        """
        dens=d0*pow(x,-2)*pow(np.tanh(y),n)* ( x*pow(1+x,-2) + n*rs/rc*pow(np.tanh(y),-1)*pow(np.sech(y),2) * (np.log(1+x)+x*pow(1+x,-1)) ) with x=m/rs, y=m/rc
        :param d0:
        :param rs:
        :param rc:
        :param n:
        :param e:
        :param mcut:
        """

        #if (n>1) or (n<=0):
            #raise ValueError('n must be 0<n<=1')

        super(core_NFW_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        self.rs=rs
        self.rc=rc
        self.n=n
        self.name='CoreNFW halo'

    def _potential_serial(self, R, Z, grid=False, toll=1e-4, mcut=None):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        self.set_toll(toll)

        return potential_core_nfw(R, Z, d0=self.d0, rs=self.rs, rc=self.rc, n=self.n, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None, nproc=2):
        """Calculate the potential in R and Z using a parallelized code.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        self.set_toll(toll)


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_core_nfw)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rs,self.rc,self.n,self.e, mcut,self.toll,grid),_sorted='sort')

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rs, self.rc, self.n, self.e, mcut, self.toll, grid),_sorted='input')


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_core_nfw(R, self.d0, self.rs, self.rc, self.n, self.e, toll=self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_core_nfw)

        htab=pardo.run_grid(R,args=(self.d0, self.rs, self.rc, self.n, self.e, self.toll),_sorted='input')

        return htab

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rs
        y = m / self.rc

        num  = self.d0

        return self.d0*pow(x,-2)*pow(np.tanh(y),self.n)* ( x*pow(1+x,-2) + self.n*self.rs/self.rc*pow(0.5*np.sinh(2*y),-1) * (np.log(1+x)-x*pow(1+x,-1)) )

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: core NFW %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f kpc \n'%self.rs
        s+='rc: %.1f\n'%self.rc
        s+='n: %.1f\n'%self.n
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s

