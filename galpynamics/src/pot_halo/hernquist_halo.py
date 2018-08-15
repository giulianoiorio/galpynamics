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
from .alfabeta_halo import alfabeta_halo


class hernquist_halo(alfabeta_halo):

    def __init__(self,d0,rs,e=0,mcut=100):
        """
        dens=d0/( (x) * (1+x)^(3))
        :param d0:
        :param rs:
        :param e:
        :param mcut:
        """

        alfa=1
        beta=4
        super(hernquist_halo,self).__init__(d0=d0,rs=rs,alfa=alfa,beta=beta,e=e,mcut=mcut)
        self.name='Hernquist halo'

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f kpc \n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f kpc \n'%self.mcut

        return s
