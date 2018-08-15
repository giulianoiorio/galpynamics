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


class alfabeta_halo(halo):

    def __init__(self,d0,rs,alfa,beta,e=0,mcut=100):
        """
        dens=d0/( (x^alfa) * (1+x)^(beta-alfa))
        :param d0:
        :param rs:
        :param alfa:
        :param beta:
        :param e:
        :param mcut:
        """

        if alfa>=2:
            raise ValueError('alpha must be <2')

        self.rs=rs
        self.alfa=alfa
        self.beta=beta
        super(alfabeta_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        self.name='AlfaBeta halo'

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

        return potential_alfabeta(R, Z, d0=self.d0, alfa=self.alfa, beta=self.beta, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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
        pardo.set_func(potential_alfabeta)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.alfa,self.beta,self.rc,self.e, mcut,self.toll,grid),_sorted='sort')

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.alfa, self.beta, self.rc, self.e, mcut, self.toll, grid),_sorted='input')


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_alfabeta(R, self.d0, self.rc, self.alfa, self.beta, self.e, toll=self.toll))

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

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.alfa, self.beta, self.e, self.toll),_sorted='input')

        return htab

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rs

        num  = self.d0
        denA = x**self.alfa
        denB = (1+x)**(self.beta-self.alfa)
        den=denA*denB

        return num / den

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f kpc \n'%self.rs
        s+='alfa: %.1f\n'%self.alfa
        s+='beta: %.1f\n'%self.beta
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s

