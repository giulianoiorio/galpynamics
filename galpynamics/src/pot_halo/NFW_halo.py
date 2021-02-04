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
from .pot_halo import halo
import numpy as np
import sys


class NFW_halo(halo):

    def __init__(self,d0,rs,e=0,mcut=100):
        """NFW halo d=d0/((r/rs)(1+r/rs)^2)

        :param d0:  Central density in Msun/kpc^3
        :param rs:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.rs=rs
        super(NFW_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        self.name='NFW halo'

    @classmethod
    def cosmo(cls, c, V200, H=67, e=0, mcut=100):
        """

        :param c:
        :param V200:  km/s
        :param H:  km/s/Mpc
        :return:
        """

        num=14.93*(V200/100.)
        den=(c/10.)*(H/67.)
        rs=num/den

        rho_crit=8340.*(H/67.)*(H/67.)
        lc=np.log(1+c)
        denc=c/(1+c)
        delta_c=(c*c*c) / (lc - denc)
        d0=rho_crit*delta_c


        return cls(d0=d0, rs=rs, e=e, mcut=mcut)

    @classmethod
    def cosmoM(cls, c, M200, H=67, e=0, mcut=100):
        """

        :param c:
        :param M200:  Msun
        :param H:  km/s/Mpc
        :return:
        """


        rho_crit=8340.*(H/67.)*(H/67.)
        lc=np.log(1+c)
        denc=c/(1+c)
        delta_c=(c*c*c) / (lc - denc)
        d0=rho_crit*delta_c

        num=3*M200
        den=(c*c*c)*rho_crit*(800*np.pi)
        rs = num / den


        return cls(d0=d0, rs=rs, e=e, mcut=mcut)


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


        return  potential_nfw(R, Z, d0=self.d0, rs=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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
        pardo.set_func(potential_nfw)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e, mcut,self.toll,grid),_sorted='sort')

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid),_sorted='input')


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_nfw(R, self.d0, self.rc, self.e, toll=self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_nfw)

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll),_sorted='input')

        return htab

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rs

        num = self.d0
        den = (x)*(1 + x)*(1 + x)

        return num / den


    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: NFW halo\n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f kpc \n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s
        