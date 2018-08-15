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


class plummer_halo(halo):

    def __init__(self,rc,d0=None,mass=None,e=0,mcut=100):
        """
        dens=d0/((1+(r/rs)^2)^-5/2)
        :param rc:
        :param d0:
        :param mass:
        :param e:
        :param mcut:
        """

        cost_mass=(4./3.)*np.pi*rc*rc*rc*(np.sqrt(1-e*e))

        if (d0 is None) and (mass is None):
            raise ValueError('d0 or mass must be set')
        elif mass is None:
            mass=d0*cost_mass
        else:
            d0=mass/cost_mass

        super(plummer_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)
        self.name='Plummer halo'
        self.mass=mass

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

        return potential_plummer(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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
        pardo.set_func(potential_plummer)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0, self.rc,self.e, mcut,self.toll,grid),_sorted='sort')

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

        return np.array(vcirc_plummer(R, self.d0, self.rc, self.e, toll=self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_plummer)

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll),_sorted='input')

        return htab

    def _mass(self,m):

        raise NotImplementedError()

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rc

        num = self.d0
        den = (1 + x * x)**(2.5)

        return num / den


    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='Mass: %.2e Msun \n'%self.mass
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f kpc \n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s
