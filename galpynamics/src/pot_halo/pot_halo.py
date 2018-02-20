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



class halo(object):
    """
    Super class for halo potentials
    """
    def __init__(self,d0,rc,e=0,mcut=100):
        """Init

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.d0=d0
        self.rc=rc
        self.e=e
        self.toll=1e-4
        self.mcut=mcut
        self.name='General halo'

    def set_toll(self,toll):
        """Set tollerance for quad integration

        :param toll: tollerance for quad integration
        :return:
        """

        self.toll=toll

    def set_mcut(self,mcut):
        """Set mcut

        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        self.mcut=mcut

    def potential(self,R,Z=0,grid=False,toll=1e-4,mcut=None, nproc=1,output='1D'):
        """Calculate potential at coordinate (R,Z). If R and Z are arrays with unequal lengths or
            if grid is True, the potential will be calculated in a 2D grid in R and Z.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :param nproc: Number of processes
        :return:  An array with:
            0-R
            1-Z
            2-Potential
        """

        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for halo.potential'%str(output))

        if isinstance(R,float) or isinstance(R, int): R=np.array((R,))
        if isinstance(Z,float) or isinstance(Z, int): Z=np.array((Z,))




        if len(R) != len(Z) or grid == True:

            if len(R) != len(Z) and grid == False:
                print('\n*WARNING*: estimate potential on model %s. \n'
                      'R and Z have different dimensions, but grid=False. R and Z will be sorted and grid set to True.\n'%self.name)
                sys.stdout.flush()

            ndim = len(R) * len(Z)
            R=np.sort(R)
            Z=np.sort(Z)
            grid=True
        else:
            ndim = len(R)
            if Dgrid==True: raise ValueError('Cannot use output 2D with non-grid input')


        if mcut is None:
            mcut=self.mcut
        else:
            self.mcut=mcut

        if nproc==1 or ndim<100000:
            ret_array = self._potential_serial(R=R,Z=Z,grid=grid,toll=toll,mcut=mcut)
        else:
            ret_array = self._potential_parallel(R=R, Z=Z, grid=grid, toll=toll, mcut=mcut,nproc=nproc)

        if grid and Dgrid:

            ret_Darray=np.zeros((3,len(R),len(Z)))
            ret_Darray[0,:,:]=ret_array[:,0].reshape(len(R),len(Z))
            ret_Darray[1,:,:]=ret_array[:,1].reshape(len(R),len(Z))
            ret_Darray[2,:,:]=ret_array[:,2].reshape(len(R),len(Z))

            return ret_Darray

        else:

            return ret_array


    def _potential_serial(self,R,Z,grid=False,toll=1e-4,mcut=None):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        raise NotImplementedError('Potential serial not implemented for this class')

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None,nproc=2):
        """Calculate the potential in R and Z using a parallelized code.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """
        raise NotImplementedError('Potential parallel not implemented for this class')

    def vcirc(self, R, toll=1e-4, nproc=1):
        """Calculate Vcirc at planare radius coordinate R.
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:  An array with:
            0-R
            1-Vcirc
        """

        ndim=len(R)
        if nproc==1 or ndim<100000:
            return self._vcirc_serial(R=R,toll=toll)
        else:
            return self._vcirc_parallel(R=R, toll=toll, nproc=nproc)

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """

        raise NotImplementedError('Vcirc serial not implemented for this class')


    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """
        raise NotImplementedError('Potential parallel not implemented for this class')


    def dens(self, R, Z=None, grid=False,output='1D'):
        """
        Evaulate the density at the point (R,Z)
        :param R: float int or iterable
        :param z: float int or iterable, if Z is None R=m elliptical radius (m=sqrt(R*R+Z*Z/(1-e^2)) if e=0 spherical radius)
        :param grid:  if True calculate the potential in a 2D grid in R and Z, if len(R)!=len(Z) grid is True by default
        :return:  2D array with: col-0 R, col-1 dens(m) if Z is None or col-0 R, col-1 Z, col-2 dens(R,Z)
        """

        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for halo.dens'%str(output))

        if isinstance(R, int) or isinstance(R, float):  R = np.array([R, ])

        if Z is not None:

            if isinstance(Z, int) or isinstance(Z, float):  Z = np.array([Z, ])


            if grid==True or len(R)!=len(Z):

                if len(R) != len(Z) and grid == False:
                    print('\n*WARNING*: estimate potential on model %s. \n'
                          'R and Z have different dimensions, but grid=False. R and Z will be sorted and grid set to True.\n' % self.name)
                    sys.stdout.flush()
                    R=np.sort(R)
                    Z=np.sort(Z)
                    grid=True

                ret=np.zeros(shape=(len(R)*len(Z),3))

                coord=cartesian(R,Z)
                ret[:,:2]=coord
                ret[:,2]=self._dens(coord[:,0],coord[:,1])

                if Dgrid:

                    ret_Darray = np.zeros((3, len(R), len(Z)))
                    ret_Darray[0, :, :] = ret[:, 0].reshape(len(R), len(Z))
                    ret_Darray[1, :, :] = ret[:, 1].reshape(len(R), len(Z))
                    ret_Darray[2, :, :] = ret[:, 2].reshape(len(R), len(Z))

                    ret=ret_Darray


            else:
                if Dgrid == True:
                    raise ValueError('Cannot use output 2D with non-grid input')
                ret=np.zeros(shape=(len(R),3))
                ret[:,0]=R
                ret[:,1]=Z
                ret[:,2]=self._dens(R,Z)

        else:

            ret=np.zeros(shape=(len(R),2))

            ret[:,0]=R
            ret[:,1]=self._dens(R)

        return ret

    def mmass(self, m):

        return self._mass(m)

    def __str__(self):

        s=''
        s+='Model: General halo\n'
        s+='d0: %.2f Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

#TODO: la vcirc dell alone isotermo e analitica per ogni e, implementare la formula nella mia tesi
class isothermal_halo(halo):

    def __init__(self,d0,rc,e=0,mcut=100):
        """Isothermal halo d=d0/(1+r^2/rc^2)

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """
        super(isothermal_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)
        self.name='Isothermal halo'

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


        return  potential_iso(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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
        pardo.set_func(potential_iso)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_iso(R, self.d0, self.rc, self.e, toll=self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_iso)

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll))

        return htab

    def _dens(self, R, Z=0):

        q2=1-self.e*self.e

        m=np.sqrt(R*R+Z*Z/q2)

        x=m/self.rc

        num=self.d0
        den=(1+x*x)

        return num/den

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: Isothermal halo\n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

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

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


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

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll))

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
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

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

            htab=pardo.run_grid(R,args=(Z,self.d0,self.alfa,self.beta,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.alfa, self.beta, self.rc, self.e, mcut, self.toll, grid))


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

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.alfa, self.beta, self.e, self.toll))

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
        s+='rs: %.2f\n'%self.rs
        s+='alfa: %.1f\n'%self.alfa
        s+='beta: %.1f\n'%self.beta
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

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
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class deVacouler_like_halo(alfabeta_halo):

    def __init__(self,d0,rs,e=0,mcut=100):
        """
        Approximation of the 3D deprojection of the R^(1/4) law (De Vacouler) with alfa-beta model dens=d0/( x^(3/2) * (1+x)^(5/2) )
        :param d0:
        :param rs:
        :param e:
        :param mcut:
        """
        alfa=1.5
        beta=4
        super(deVacouler_like_halo, self).__init__(d0=d0, rs=rs, alfa=alfa, beta=beta, e=e, mcut=mcut)
        self.name = 'deVacouler like halo'


    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

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
        pardo.set_func(potential_plummer())

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0, self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


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

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll))

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
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

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

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc, self.n, self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.n, self.e, mcut, self.toll, grid))
        cpdef

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

        htab=pardo.run_grid(R,args=(self.d0, self.rs, self.n, self.e,self.toll))

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
        s+='rs: %.2f\n'%self.rc
        s+='n: %.2f\n'%self.n
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class valy_halo(halo):

    def __init__(self,rb,d0=None,mass=None,e=0,mcut=100):
        """
        dens=d0*exp(-m^2/rb^2)
        where d0=mass/((2 pi)^(3/2) r_b^3)
        one between d0 and mass must be set, d0 has the priority
        :param rb: Scale radius
        :param d0: central density in Msun/kpc^3
        :param mass: total mass in Msun
        :param e: eccentricity (e=0, spherical)
        :param mcut:  elliptical radius where dens(m>mcut)=0
        """

        cost_mass=((2*np.pi)**1.5)*(rb*rb*rb)*(np.sqrt(1-e*e))

        if (d0 is None) and (mass is None):
            raise ValueError('d0 or mass must be set')
        elif mass is None:
            mass=d0*cost_mass
        else:
            d0=mass/cost_mass

        super(valy_halo,self).__init__(d0=d0,rc=rb,e=e,mcut=mcut)
        self.name='Valy halo'
        self.mass=mass
        self.rb=rb

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

        return potential_valy(R, Z, d0=self.d0, rb=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)


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
        pardo.set_func(potential_valy())

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0, self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_valy(R, self.d0, self.rc, self.e, toll=self.toll))


    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_valy)

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll))

        return htab

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rc



        return self.d0*np.exp(-0.5*x*x)

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='Mass: %.2e Msun \n'%self.mass
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rb: %.2f kpc\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class exponential_halo(halo):

    def __init__(self,rb,d0=None,mass=None,e=0,mcut=100):
        """
        dens=d0*exp(-m/rb)
        where d0=mass/(8*PI*rb^3)
        one between d0 and mass must be set, d0 has the priority
        :param rb: Scale radius
        :param d0: central density in Msun/kpc^3
        :param mass: total mass in Msun
        :param e: eccentricity (e=0, spherical)
        :param mcut:  elliptical radius where dens(m>mcut)=0
        """

        cost_mass=8*np.pi*rb*rb*rb*(np.sqrt(1-e*e))

        if (d0 is None) and (mass is None):
            raise ValueError('d0 or mass must be set')
        elif mass is None:
            mass=d0*cost_mass
        else:
            d0=mass/cost_mass

        super(exponential_halo,self).__init__(d0=d0,rc=rb,e=e,mcut=mcut)
        self.name='Exponential halo'
        self.mass=mass
        self.rb=rb

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

        return potential_exponential(R, Z, d0=self.d0, rb=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)


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
        pardo.set_func(potential_exponential())

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0, self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_exponential(R, self.d0, self.rc, self.e, toll=self.toll))


    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_exponential)

        htab=pardo.run_grid(R,args=(self.d0, self.rc, self.e, self.toll))

        return htab

    def _dens(self, R, Z=0):

        q2 = 1 - self.e * self.e

        m = np.sqrt(R * R + Z * Z / q2)

        x = m / self.rc



        return self.d0*np.exp(-x)

    def _mass(self,m):

        raise NotImplementedError()

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='Mass: %.2e Msun \n'%self.mass
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rb: %.2f kpc\n'%self.rb
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s