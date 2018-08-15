from __future__ import division, print_function
from .pot_c_ext.isothermal_halo import potential_iso,  vcirc_iso
from .pot_c_ext.nfw_halo import potential_nfw, vcirc_nfw
from .pot_c_ext.alfabeta_halo import potential_alfabeta, vcirc_alfabeta
from .pot_c_ext.plummer_halo import potential_plummer, vcirc_plummer
from .pot_c_ext.einasto_halo import potential_einasto, vcirc_einasto
from .pot_c_ext.valy_halo import potential_valy, vcirc_valy
from .pot_c_ext.exponential_halo import potential_exponential, vcirc_exponential
import multiprocessing as mp
from ..pardo.Pardo import ParDo
from ..utility import cartesian
import numpy as np
import sys
from .pot_halo import halo


class einasto_halo(halo):

    def __init__(self,d0,rs,n,e=0,mcut=100):
        """einasto halo d=d0*exp(-dn*(r/rs)^(1/n))

        :param d0:  Central density in Msun/kpc^3
        :param rs:  Scale radius in kpc
        :param n:
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.rs=rs
        super(einasto_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        dnn=self.dn(n)
        self.de=self.d0/np.exp(dnn)
        self.n=n
        self.name='Einasto halo'

    @classmethod
    def de(cls,de,rs,n,e=0,mcut=100):

        dnn=cls.dn(n)
        d0=de*np.exp(dnn)

        return cls(d0=d0, rs=rs, n=n, e=e, mcut=mcut)



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


        return  potential_einasto(R, Z, d0=self.d0, rs=self.rc, n=self.n, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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
        pardo.set_func(potential_einasto)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc, self.n, self.e, mcut,self.toll,grid),_sorted='sort')

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.n, self.e, mcut, self.toll, grid),_sorted='input')
        

        return htab

    @staticmethod
    def dn(n):

        n2=n*n
        n3=n2*n
        n4=n3*n
        a0=3*n
        a1 = -1. / 3.
        a2 = 8. / (1215. * n)
        a3 = 184. / (229635. * n2)
        a4 = 1048 / (31000725. * n3)
        a5 = -17557576 / (1242974068875. * n4)

        return a0+a1+a2+a3+a4+a5

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_einasto(R, self.d0, self.rs, self.n, self.e,self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_alfabeta)

        htab=pardo.run_grid(R,args=(self.d0, self.rs, self.n, self.e,self.toll),_sorted='input')

        return htab



    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rc

        dnn=self.dn(self.n)

        A=x**(1/self.n)

        ret=self.d0*np.exp(-dnn*A)

        return ret

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='de: %.2e Msun/kpc3 \n'%self.de
        s+='rs: %.2f kpc \n'%self.rc
        s+='n: %.2f\n'%self.n
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s
