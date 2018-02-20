from __future__ import division, print_function
from .pot_c_ext.integrand_functions import potential_disc, potential_disc_thin
from .pot_c_ext.integrand_vcirc import vcirc_disc, vcirc_disc_thin
from .pot_c_ext.rflare_law import flare as flare_func
from .pot_c_ext.rdens_law import rdens as rdens_func
from .pot_c_ext.zdens_law import hwhm_fact, zfunc_dict
import multiprocessing as mp
from ..pardo.Pardo import ParDo
import  numpy as np
from scipy.optimize import curve_fit
import emcee
from .pot_c_ext.model_option import checkfl_dict, checkrd_dict
from scipy.integrate import quad, nquad
import sys
from ..utility import cartesian





def _fit_utility(f,rfit_array,p0):


    if rfit_array.shape[1] == 2:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = None
    elif rfit_array.shape[1] == 3:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = rfit_array[:, 2]
    else:
        raise ValueError('Wrong rfit dimension')

    popt, pcov = curve_fit(f=f, xdata=R, ydata=Sigma, sigma=Sigma_err, absolute_sigma=True, p0=p0)

    return popt, pcov

def _fit_utility_poly(degree,rfit_array):


    if rfit_array.shape[1] == 2:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = None
    elif rfit_array.shape[1] == 3:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = 1/rfit_array[:, 2]
    else:
        raise ValueError('Wrong rfit dimension')

    popt = np.polyfit(R,Sigma,deg=degree,w=Sigma_err)


    return popt[::-1], 0

#########PolyExp
def _funco(x,R):

    Rd= x[0]
    rcoeff = x[1:]
    p = np.poly1d(rcoeff[::-1])
    yobs = np.exp(-R / Rd) * p(R)

    return yobs

def _lnprob_halo(x, R, yteo, yerr):

    if x[0]<0:
        return -np.inf

    yobs=_funco(x,R)

    if yerr is None:
        yerr=1

    return -np.sum(((yobs - yteo) * (yobs - yteo))/(yerr*yerr) )

def _fit_utility_rpoly(degree,rfit_array,nproc=1):

    if rfit_array.shape[1] == 2:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = None
    elif rfit_array.shape[1] == 3:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = rfit_array[:, 2]
    else:
        raise ValueError('Wrong rfit dimension')

    if degree>8:
        raise ValueError('Maximum degree is 8')

    x0 = [np.mean(R)/2.,Sigma[0]]+list(np.zeros(degree-1))
    ndim, nwalkers = degree+1, 300

    pos = [x0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob_halo, args=(R, Sigma, Sigma_err),threads=nproc)
    sampler.run_mcmc(pos, 500)

    samples = sampler.flatchain[100:]
    postprob = sampler.flatlnprobability[100:]
    maxlik_idx = np.argmax(postprob)
    best_pars = samples[maxlik_idx, :]
    best_like = postprob[maxlik_idx]



    return best_pars, best_like, samples
###########

######FRATLAW
def _funco_fratlaw(x,R):

    s0, Rd, Rd2, alpha = x
    yobs = s0*np.exp(-R/Rd)*(1+(R/Rd2))**(alpha)

    return yobs

	
	
def _lnprob_halo_fratlaw(x, R, yteo, yerr):

    s0, Rd, Rd2, alpha = x
    if s0<0 or Rd<0 or Rd2<0:
        return -np.inf

    yobs=_funco_fratlaw(x,R)

    if yerr is None:
        yerr=1
    
    chi2=-np.sum(((yobs - yteo) * (yobs - yteo))/(yerr*yerr) )
    if np.isfinite(chi2):

        return -np.sum(((yobs - yteo) * (yobs - yteo))/(yerr*yerr) )
    else:
        return -np.inf


def _fit_utility_fratlaw(rfit_array,nproc=1):

    if rfit_array.shape[1] == 2:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = None
    elif rfit_array.shape[1] == 3:
        R = rfit_array[:, 0]
        Sigma = rfit_array[:, 1]
        Sigma_err = rfit_array[:, 2]
    else:
        raise ValueError('Wrong rfit dimension')



    x0 = [rfit_array[0,1], np.mean(rfit_array[:,0]), np.mean(rfit_array[:,0]), 1]
    ndim, nwalkers = 4, 300

    pos = [x0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob_halo_fratlaw, args=(R, Sigma, Sigma_err),threads=nproc)
    sampler.run_mcmc(pos, 500)

    samples = sampler.flatchain[100:]
    postprob = sampler.flatlnprobability[100:]
    maxlik_idx = np.argmax(postprob)
    best_pars = samples[maxlik_idx, :]
    best_like = postprob[maxlik_idx]


    return best_pars, best_like, samples
###########


class disc(object):
    """
    Super class for halo potentials
    """
    def __init__(self,sigma0,rparam,fparam,zlaw,rlaw,flaw,Rcut=50,zcut=30):
        """Init

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.sigma0=sigma0
        self.rparam=np.zeros(10)
        self.fparam=np.zeros(10)
        self.zlaw=zlaw
        self.zfunc=zfunc_dict[zlaw]
        self.rlaw=rlaw
        self.flaw=flaw
        self.lenparam = 10
        self.Rcut= Rcut
        self.zcut= zcut
        self.Rlimit = None
        self.name ='General disc'


        #print('LPARAM',rparam)
        #print(type(rparam[0]))
        #Make rparam
        lrparam=len(rparam)
        if lrparam>self.lenparam: raise ValueError('rparam length cannot exced %i'%self.lenparam)
        elif lrparam<self.lenparam: self.rparam[:lrparam]=rparam
        else: self.rparam[:]=rparam

        #Make fparam
        lfparam=len(fparam)
        if lfparam>self.lenparam: raise ValueError('fparam length cannot exced %i'%self.lenparam)
        elif lfparam<self.lenparam: self.fparam[:lfparam]=fparam
        else: self.fparam[:]=fparam

        if zlaw=='dirac':
            self._pot_serial    =   self._potential_serial_thin
            self._pot_parallel  =   self._potential_parallel_thin
            self._vc_serial     =   self._vcirc_serial_thin
            self._vc_parallel   =   self._vcirc_parallel_thin
            self.flare         =    self._flare_thin
        else:
            self._pot_serial    =   self._potential_serial
            self._pot_parallel  =   self._potential_parallel
            self._vc_serial     =   self._vcirc_serial
            self._vc_parallel   =   self._vcirc_parallel
            self.flare          =   self._flare


    def potential(self,R,Z,grid=False,toll=1e-4,Rcut=None, zcut=None, nproc=1, output='1D'):
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

        if Rcut is None:
            Rcut=self.Rcut
        else:
            self.Rcut=Rcut

        if zcut is None:
            zcut=self.zcut
        else:
            self.zcut=zcut


        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for disc.potential'%str(output))

        if isinstance(R,float) or isinstance(R, int): R=np.array((R,))
        if isinstance(Z,float) or isinstance(Z, int): Z=np.array((Z,))



        if len(R) != len(Z) or grid == True:

            if len(R) != len(Z) and grid == False:
                print('\n*WARNING*: estimate potential on model %s. \n'
                      'R and Z have different dimensions, but grid=False. R and Z will be sorted and grid set to True.\n'%self.name)
                sys.stdout.flush()

            R=np.sort(R)
            Z=np.sort(Z)
            grid=True

        else:

            if Dgrid==True: raise ValueError('Cannot use output 2D with non-grid input')



        if nproc==1:
            ret_array= self._pot_serial(R=R,Z=Z,grid=grid,toll=toll,Rcut=Rcut,zcut=zcut)
        else:
            ret_array = self._pot_parallel(R=R, Z=Z, grid=grid, toll=toll, Rcut=Rcut, zcut=zcut, nproc=nproc)

        if grid and Dgrid:

            ret_Darray=np.zeros((3,len(R),len(Z)))
            ret_Darray[0,:,:]=ret_array[:,0].reshape(len(R),len(Z))
            ret_Darray[1,:,:]=ret_array[:,1].reshape(len(R),len(Z))
            ret_Darray[2,:,:]=ret_array[:,2].reshape(len(R),len(Z))

            return ret_Darray

        else:

            return ret_array




    def _potential_serial(self,R,Z,grid=False,toll=1e-4,Rcut=None, zcut=None, **kwargs):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)

        if zcut is not None: zcut=zcut
        elif (isinstance(Z,float) or isinstance(Z, int)): zcut=10*Z
        else: zcut=10*np.max(Z)


        return potential_disc(R,Z,sigma0=self.sigma0, rcoeff=self.rparam, fcoeff=self.fparam,zlaw=self.zlaw, rlaw=self.rlaw, flaw=self.flaw,rcut=Rcut,zcut=zcut, toll=toll, grid=grid)


    def _potential_serial_thin(self,R,Z,grid=False,toll=1e-4,Rcut=None, **kwargs):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)


        return potential_disc_thin(R,Z,sigma0=self.sigma0, rcoeff=self.rparam, rlaw=self.rlaw, rcut=Rcut, toll=toll, grid=grid)



    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, Rcut=None, zcut=None, nproc=2, **kwargs):

        '''
        if Rcut is not None:
            Rcut = Rcut
        elif (isinstance(R, float) or isinstance(R, int)):
            Rcut = 3 * R
        else:
            Rcut = 3 * np.max(R)

        if zcut is not None:
            zcut = zcut
        elif (isinstance(Z, float) or isinstance(Z, int)):
            zcut = 10 * Z
        else:
            zcut = 10 * np.max(Z)
        '''


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_disc)

        R=np.sort(R)
        Z=np.sort(Z)

        if len(R)!=len(Z) or grid==True:

            htab = pardo.run_grid(R,args=(Z,self.sigma0,self.rparam,self.fparam,self.zlaw,self.rlaw,self.flaw, Rcut, zcut, toll,grid))

        else:

            htab = pardo.run(R, Z, args=(self.sigma0, self.rparam, self.fparam, self.zlaw, self.rlaw, self.flaw, Rcut, zcut, toll, grid))


        return htab

    def _potential_parallel_thin(self, R, Z, grid=False, toll=1e-4, Rcut=None,  nproc=2, **kwargs):

        '''
        if Rcut is not None:
            Rcut = Rcut
        elif (isinstance(R, float) or isinstance(R, int)):
            Rcut = 3 * R
        else:
            Rcut = 3 * np.max(R)
        '''


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_disc_thin)

        if len(R)!=len(Z) or grid==True:

            htab = pardo.run_grid(R,args=(Z,self.sigma0,self.rparam,self.rlaw, Rcut, toll, grid))

        else:

            htab = pardo.run(R, Z, args=(self.sigma0,self.rparam,self.rlaw, Rcut, toll, grid))


        return htab


    def vcirc(self, R, toll=1e-4,Rcut=None, zcut=None, nproc=1):
        """

        :param R:
        :param toll:
        :param Rcut:
        :param zcut:
        :param nproc:
        :return:
        """
        if Rcut is None:
            Rcut=self.Rcut
        else:
            self.Rcut=Rcut

        if zcut is None:
            zcut=self.zcut
        else:
            self.zcut=zcut

        if nproc==1:
            return self._vc_serial(R=R,toll=toll,Rcut=Rcut,zcut=zcut)
        else:
            return self._vc_parallel(R=R, toll=toll, Rcut=Rcut, zcut=zcut, nproc=nproc)


    def _vcirc_serial(self, R, toll=1e-4, Rcut=None, zcut=None, **kwargs):

        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)

        if zcut is not None: zcut=zcut
        elif (isinstance(R,float) or isinstance(R, int)): zcut=10*R
        else: zcut=10*np.max(R)


        return vcirc_disc(R=R, sigma0=self.sigma0, rcoeff=self.rparam, fcoeff=self.fparam, zlaw=self.zlaw, rlaw=self.rlaw, flaw=self.flaw, rcut=Rcut, zcut=zcut, toll=toll)


    def _vcirc_serial_thin(self, R, toll=1e-4, Rcut=None, **kwargs):

        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)


        return vcirc_disc_thin(R=R, sigma0=self.sigma0, rcoeff=self.rparam, rlaw=self.rlaw,  rcut=Rcut, toll=toll)

    def _vcirc_parallel(self, R, toll=1e-4, Rcut=None, zcut=None, nproc=2, **kwargs):


        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_disc)

        htab = pardo.run_grid(R, args=(self.sigma0, self.rparam, self.fparam, self.zlaw, self.rlaw, self.flaw, Rcut, zcut, toll))

        return htab

    def _vcirc_parallel_thin(self, R, toll=1e-4, Rcut=None,  nproc=2, **kwargs):


        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_disc_thin)

        htab = pardo.run_grid(R, args=(self.sigma0, self.rparam, self.rlaw, Rcut, toll))

        return htab

    def _flare(self, R, HWHM=False):

        checkfli=checkfl_dict[self.flaw]

        ret = flare_func(R, checkfli, *self.fparam)

        if HWHM:
            zd_to_HWHM = hwhm_fact[self.zlaw]
            ret[:,1]=ret[:,1]*zd_to_HWHM

        return ret

    def _flare_thin(self, R, **kwargs):


        if isinstance(R, int) or isinstance(R, float):
            return np.array([[R,0]])
        elif isinstance(R, list) or isinstance(R, float) or isinstance(R, np.ndarray):
            ret=np.zeros(shape=(len(R),2))
            ret[:,0]=R
            return ret

    def Sdens(self, R):
        """
        Disc surface density
        :param R: cylindrical radius
        :return:
        """

        checkrdi=checkrd_dict[self.rlaw]
        sdens=rdens_func(R, checkrdi, *self.rparam)
        sdens[:,1]=self.sigma0*sdens[:,1]

        return sdens


    def dens(self, R, Z=0, grid=False, output='1D'):
        """
        Evaulate the density at the point (R,Z)
        :param R: float int or iterable
        :param z: float int or iterable, if Z is None R=m elliptical radius (m=sqrt(R*R+Z*Z/(1-e^2)) if e=0 spherical radius)
        :param grid:  if True calculate the potential in a 2D grid in R and Z, if len(R)!=len(Z) grid is True by default
        :return:  2D array with: col-0 R, col-1 dens(m) if Z is None or col-0 R, col-1 Z, col-2 dens(R,Z)
        """

        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for disc.dens'%str(output))

        if isinstance(R, int) or isinstance(R, float):  R = np.array([R, ])
        if isinstance(Z, int) or isinstance(Z, float):  Z = np.array([Z, ])



        if grid==True or len(R)!=len(Z):


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


            Sdens = self.Sdens(coord[:,0])[:, 1]
            zd = self.flare(coord[:,0])[:, 1]
            gzd = self.zfunc(coord[:,1], zd)



        else:
            if Dgrid==True:
                raise ValueError('Cannot use output 2D with non-grid input')

            ret=np.zeros(shape=(len(R),3))
            ret[:,0]=R
            ret[:,1]=Z

            Sdens = self.Sdens(R)[:, 1]
            zd = self.flare(R)[:, 1]
            gzd = self.zfunc(Z, zd)

        ret[:, 2] = Sdens * gzd

        if grid and Dgrid:
            ret_Darray = np.zeros((3, len(R), len(Z)))
            ret_Darray[0, :, :] = ret[:, 0].reshape(len(R), len(Z))
            ret_Darray[1, :, :] = ret[:, 1].reshape(len(R), len(Z))
            ret_Darray[2, :, :] = ret[:, 2].reshape(len(R), len(Z))

            ret = ret_Darray

        return ret


    def diskmass(self,up,low=0):
        int_func=lambda r: (self.Sdens(r)[:,1])*(2*np.pi*r)
        int=quad(int_func,low,up)[0]
        return int

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2e Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Flaring law: %s \n'%self.flaw
        s+='Rparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.rparam)
        s+='Fparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.fparam)
        s+='Rcut: %.3f kpc \n'%self.Rcut
        s+='zcut: %.3f kpc \n'%self.zcut
        if self.Rlimit is None:
            s+='Rlimit: None \n'
        else:
            s+='Rlimit: %.3f kpc \n'%self.Rlimit
        return s


class Exponential_disc(disc):

    def __init__(self,sigma0,Rd,fparam,zlaw='gau',flaw='poly',Rcut=50, zcut=30):

        rparam=np.array([Rd,1])
        self.Rd=Rd

        super(Exponential_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='epoly',flaw=flaw,Rcut=Rcut, zcut=zcut)
        self.name='Exponential disc'


    @classmethod
    def thin(cls,sigma0=None,Rd=None,rfit_array=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, Rd: s0*np.exp(-R/Rd)
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0]))
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,Rd=popt
        elif (sigma0 is not None) and (Rd is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,Rd=Rd,fparam=fparam,zlaw='dirac',flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def thick(cls,sigma0=None, Rd=None, zd=None, rfit_array=None, ffit_array=None, zlaw='gau', Rcut=50, zcut=30,check_thin=True,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, Rd: s0*np.exp(-R/Rd)
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0]))
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,Rd=popt
        elif (sigma0 is not None) and (Rd is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit=lambda R,zd: np.where(R==0,zd,zd)
            p0=(np.median(ffit_array[:,1]),)
            popt,pcov=_fit_utility(func_fit,ffit_array,p0)
            zd=popt[0]
        elif (zd is not None):
            pass
        else:
            raise ValueError()


        if check_thin:

            if zd<0.01:
                print('Warning Zd lower than 0.01, switching to thin disc')
                fparam=np.array([0,0])
                zlaw='dirac'
            else:
                fparam=np.array([zd,0])

        else:

            fparam = np.array([zd, 0])



        return cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def polyflare(cls,sigma0=None,Rd=None, polycoeff=None, rfit_array=None, ffit_array=None, fitdegree=4, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, Rd: s0*np.exp(-R/Rd)
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0]))
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,Rd=popt
        elif (sigma0 is not None) and (Rd is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            if fitdegree>7:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)' % lenp)
            else:
                popt,pcov=_fit_utility_poly(fitdegree, ffit_array)
                polycoeff=popt
                lenp = len(polycoeff)
        elif polycoeff is not None:
            lenp=len(polycoeff)
            if lenp>8:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)
        else:
            raise ValueError()

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        cls_ret=cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='poly', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret
    @classmethod
    def asinhflare(cls,sigma0=None,Rd=None, h0=None, Rf=None, c=None, rfit_array=None, ffit_array=None, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, Rd: s0*np.exp(-R/Rd)
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0]))
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,Rd=popt
        elif (sigma0 is not None) and (Rd is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.arcsinh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.arcsinh(Rlimit * Rlimit / (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='asinh',  Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def tanhflare(cls,sigma0=None,Rd=None, h0=None, Rf=None, c=None, rfit_array=None, ffit_array=None, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, Rd: s0*np.exp(-R/Rd)
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0]))
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,Rd=popt
        elif (sigma0 is not None) and (Rd is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.tanh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit / (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret= cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='tanh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None, ffit_array=None, fitdegree=None, Rlimit=None, zcut=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        Rd=self.Rd
        Rcut=self.Rcut
        if zcut is None:
            zcut=self.zcut
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw




        if flaw=='thin':
            return Exponential_disc.thin(sigma0=sigma0,Rd=Rd, Rcut=Rcut, zcut=zcut)
        elif flaw=='thick':
            if (zd is not None) or (ffit_array is not None):
                return Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd,zlaw=zlaw, ffit_array=ffit_array,Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('zd or ffit_array must be a non None value for thick flaw')
        elif flaw=='poly':
            if (polycoeff is not None) or (ffit_array is not None):
                if fitdegree is None: fitdegree=3
                return Exponential_disc.polyflare(sigma0=sigma0, Rd=Rd, polycoeff=polycoeff,zlaw=zlaw, ffit_array=ffit_array, fitdegree=fitdegree, Rlimit=Rlimit,Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('polycoeff or ffit_array must be a non None value for poly flaw')
        elif flaw=='asinh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Exponential_disc.asinhflare(sigma0=sigma0, Rd=Rd, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit,Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='tanh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Exponential_disc.tanhflare(sigma0=sigma0, Rd=Rd, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit,Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')
        else:
            raise ValueError('Flaw %s does not exist: chose between thin, thick, poly, asinh, tanh'%flaw)



    def take_radial_from(self, cls):

        if isinstance(cls, Exponential_disc)==False:
            raise ('Value Error: dynamic component mismatch, given %s required %s'%(cls.name, self.name))

        self.sigma0=cls.sigma0
        self.rparam=cls.rparam
        self.rlaw=cls.rlaw
        self.Rd=cls.Rd
        self.Rcut=cls.Rcut


    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2e Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.3f kpc \n'%self.Rd
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.fparam)
        s+='Rcut: %.3f kpc \n'%self.Rcut
        s+='zcut: %.3f kpc \n'%self.zcut
        if self.Rlimit is None:
            s+='Rlimit: None \n'
        else:
            s+='Rlimit: %.3f kpc \n'%self.Rlimit
        return s


class PolyExponential_disc(disc):

    def __init__(self,sigma0,Rd,coeff,fparam,zlaw='gau',flaw='poly',Rcut=50, zcut=30):

        if isinstance(coeff,float) or isinstance(coeff,int):
            self.coeff=[1,]
        elif coeff[0]!=0:
            coeff=np.array(coeff)
            self.coeff=coeff/coeff[0]
        else:
            print('Warning, the Surface density is 0 at R=0, Sigma0 is not the value of the centrla surface density')
            self.coeff=coeff

        if len(coeff)>8:
            raise NotImplementedError('Maximum polynomial degree is 8')




        rparam=np.array([Rd,]+list(self.coeff))
        self.Rd=Rd

        super(PolyExponential_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='epoly',flaw=flaw, Rcut=Rcut, zcut=zcut)
        self.name='PolyExponential disc'


    @classmethod
    def thin(cls,sigma0=None,Rd=None,coeff=None,rfit_array=None, rfit_degree=3,Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_rpoly(rfit_degree,rfit_array,nproc=nproc)
            Rd=popt[0]
            sigma0=popt[1]
            coeff=popt[1:]/sigma0
            print('Done')
        elif (sigma0 is not None) and (Rd is not None) and (coeff is not None):
            pass
        else:
            raise ValueError()

        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,Rd=Rd,coeff=coeff,fparam=fparam,zlaw='dirac',flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def thick(cls,sigma0=None, Rd=None, coeff=None, zd=None, rfit_array=None, rfit_degree=3, ffit_array=None, zlaw='gau',Rcut=50, zcut=30, check_thin=True,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_rpoly(rfit_degree,rfit_array,nproc=nproc)
            Rd=popt[0]
            sigma0=popt[1]
            coeff=popt[1:]/sigma0
            print('Done')
        elif (sigma0 is not None) and (Rd is not None) and (coeff is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit=lambda R,zd: np.where(R==0,zd,zd)
            p0=(np.median(ffit_array[:,1]),)
            popt,pcov=_fit_utility(func_fit,ffit_array,p0)
            zd=popt[0]
        elif (zd is not None):
            pass
        else:
            raise ValueError()

        if check_thin:
            if zd<0.01:
                print('Warning Zd lower than 0.01, switching to thin disc')
                fparam=np.array([0,0])
                zlaw='dirac'
            else:
                fparam=np.array([zd,0])
        else:
            fparam = np.array([zd, 0])

        return cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def polyflare(cls,sigma0=None, Rd=None, coeff=None, polycoeff=None, rfit_array=None, rfit_degree=3, ffit_array=None, fitdegree=4, zlaw='gau', Rlimit=None,Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_rpoly(rfit_degree,rfit_array,nproc=nproc)
            Rd=popt[0]
            sigma0=popt[1]
            coeff=popt[1:]/sigma0
            print('Done')
        elif (sigma0 is not None) and (Rd is not None) and (coeff is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            if fitdegree>7:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)' % lenp)
            else:
                popt,pcov=_fit_utility_poly(fitdegree, ffit_array)
                polycoeff=popt
                lenp = len(polycoeff)
        elif polycoeff is not None:
            lenp=len(polycoeff)
            if lenp>8:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)
        else:
            raise ValueError()

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        cls_ret=cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='poly', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def asinhflare(cls,sigma0=None, Rd=None, coeff=None, h0=None, Rf=None, c=None, rfit_array=None, rfit_degree=3, ffit_array=None, zlaw='gau', Rlimit=None,Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_rpoly(rfit_degree,rfit_array,nproc=nproc)
            Rd=popt[0]
            sigma0=popt[1]
            coeff=popt[1:]/sigma0
            print('Done')
        elif (sigma0 is not None) and (Rd is not None) and (coeff is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.arcsinh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.arcsinh(Rlimit * Rlimit / (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='asinh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def tanhflare(cls,sigma0=None, Rd=None, coeff=None, h0=None, Rf=None, c=None, rfit_array=None, rfit_degree=3, ffit_array=None, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_rpoly(rfit_degree,rfit_array,nproc=nproc)
            Rd=popt[0]
            sigma0=popt[1]
            coeff=popt[1:]/sigma0
            print('Done')
        elif (sigma0 is not None) and (Rd is not None) and (coeff is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.tanh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit/ (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='tanh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,  ffit_array=None, fitdegree=None,  Rlimit=None, zcut=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        coeff=self.coeff
        Rd=self.Rd
        Rcut= self.Rcut
        if zcut is None:
            zcut=self.zcut
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return PolyExponential_disc.thin(sigma0=sigma0,Rd=Rd, coeff=coeff, Rcut=Rcut, zcut=zcut)
        elif flaw=='thick':
            if (zd is not None) or (ffit_array is not None):
                return PolyExponential_disc.thick(sigma0=sigma0,Rd=Rd, coeff=coeff, zd=zd, zlaw=zlaw, ffit_array=ffit_array, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('zd or ffit_array must be a non None value for thick flaw')
        elif flaw=='poly':
            if (polycoeff is not None) or (ffit_array is not None):
                if fitdegree is None: fitdegree=3
                return PolyExponential_disc.polyflare(sigma0=sigma0,Rd=Rd, coeff=coeff, polycoeff=polycoeff, zlaw=zlaw, ffit_array=ffit_array, fitdegree=fitdegree, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('polycoeff or ffit_array must be a non None value for poly flaw')
        elif flaw=='asinh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return PolyExponential_disc.asinhflare(sigma0=sigma0,Rd=Rd, coeff=coeff, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='tanh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return PolyExponential_disc.tanhflare(sigma0=sigma0,Rd=Rd, coeff=coeff, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')
        else:
            raise ValueError('Flaw %s does not exist: chose between thin, thick, poly, asinh, tanh'%flaw)

    def take_radial_from(self, cls):

        if isinstance(cls, PolyExponential_disc) == False:
            raise ('Value Error: dynamic component mismatch, given %s required %s' % (cls.name, self.name))

        self.sigma0 = cls.sigma0
        self.rparam = cls.rparam
        self.rlaw = cls.rlaw
        self.Rd = cls.Rd
        self.Rcut = cls.Rcut

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2e Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.3f kpc \n'%self.Rd
        s+='Polycoeff: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.rparam[1:])
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.fparam)
        s+='Rcut: %.3f kpc \n'%self.Rcut
        s+='zcut: %.3f kpc \n'%self.zcut
        if self.Rlimit is None:
            s+='Rlimit: None \n'
        else:
            s+='Rlimit: %.3f kpc \n'%self.Rlimit
        return s

class Gaussian_disc(disc):

    def __init__(self,sigma0, sigmad, R0, fparam,zlaw='gau',flaw='poly', Rcut=50, zcut=30):

        rparam    = np.zeros(10)
        rparam[0] = sigmad
        rparam[1] = R0
        self.sigmad=sigmad
        self.R0=R0

        super(Gaussian_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='gau',flaw=flaw, Rcut=Rcut, zcut=zcut)
        self.name='Gaussian disc'


    @classmethod
    def thin(cls,sigma0=None,sigmad=None,R0=None, rfit_array=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, sigmad, R0: s0*np.exp(-0.5*( (R-R0)*(R-R0) )/(sigmad*sigmad))
            idxmax=np.argmax(rfit_array[:,1])
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0])/4,rfit_array[idxmax,0])
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,sigmad,R0=popt
        elif (sigma0 is not None) and (sigmad is not None) and (R0 is not None):
            pass
        else:
            raise ValueError()

        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,sigmad=sigmad,R0=R0,fparam=fparam,zlaw='dirac',flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def thick(cls,sigma0=None, sigmad=None, R0=None, zd=None, rfit_array=None, ffit_array=None, zlaw='gau', Rcut=50, zcut=30, check_thin=True,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, sigmad, R0: s0*np.exp(-0.5*( (R-R0)*(R-R0) )/(sigmad*sigmad))
            idxmax=np.argmax(rfit_array[:,1])
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0])/4,rfit_array[idxmax,0])
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,sigmad,R0=popt
        elif (sigma0 is not None) and (sigmad is not None) and (R0 is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit=lambda R,zd: np.where(R==0,zd,zd)
            p0=(np.median(ffit_array[:,1]),)
            popt,pcov=_fit_utility(func_fit,ffit_array,p0)
            zd=popt[0]
        elif (zd is not None):
            pass
        else:
            raise ValueError()

        if check_thin:
            if zd<0.01:
                print('Warning Zd lower than 0.01, switching to thin disc')
                fparam=np.array([0,0])
                zlaw='dirac'
            else:
                fparam=np.array([zd,0])
        else:
            fparam = np.array([zd, 0])

        return cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def polyflare(cls,sigma0=None, sigmad=None, R0=None, polycoeff=None, rfit_array=None, ffit_array=None, fitdegree=4, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, sigmad, R0: s0*np.exp(-0.5*( (R-R0)*(R-R0) )/(sigmad*sigmad))
            idxmax=np.argmax(rfit_array[:,1])
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0])/4,rfit_array[idxmax,0])
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,sigmad,R0=popt
        elif (sigma0 is not None) and (sigmad is not None) and (R0 is not None):
            pass
        else:
            raise ValueError()



        #Flaw
        if ffit_array is not None:
            if fitdegree>7:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)' % lenp)
            else:
                popt,pcov=_fit_utility_poly(fitdegree, ffit_array)
                polycoeff=popt
                lenp = len(polycoeff)
        elif polycoeff is not None:
            lenp=len(polycoeff)
            if lenp>8:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)
        else:
            raise ValueError()

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        cls_ret=cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='poly', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def asinhflare(cls,sigma0=None, sigmad=None, R0=None, h0=None, Rf=None, c=None, rfit_array=None, ffit_array=None, zlaw='gau', Rlimit=None, Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, sigmad, R0: s0*np.exp(-0.5*( (R-R0)*(R-R0) )/(sigmad*sigmad))
            idxmax=np.argmax(rfit_array[:,1])
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0])/4,rfit_array[idxmax,0])
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,sigmad,R0=popt
        elif (sigma0 is not None) and (sigmad is not None) and (R0 is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.arcsinh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.arcsinh(Rlimit * Rlimit / (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='asinh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def tanhflare(cls, sigma0=None, sigmad=None, R0=None, h0=None, Rf=None, c=None, zlaw=None, rfit_array=None, ffit_array=None, Rlimit=None, Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            func_fit=lambda R, s0, sigmad, R0: s0*np.exp(-0.5*( (R-R0)*(R-R0) )/(sigmad*sigmad))
            idxmax=np.argmax(rfit_array[:,1])
            p0=(rfit_array[0,1],np.mean(rfit_array[:,0])/4,rfit_array[idxmax,0])
            popt,pcov=_fit_utility(func_fit,rfit_array,p0)
            sigma0,sigmad,R0=popt
        elif (sigma0 is not None) and (sigmad is not None) and (R0 is not None):
            pass
        else:
            raise ValueError()



        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.tanh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit/ (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='tanh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,ffit_array=None,fitdegree=None,Rlimit=None, Rcut=50, zcut=30):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        sigmad=self.sigmad
        R0=self.R0
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return Gaussian_disc.thin(sigma0=sigma0,sigmad=sigmad, R0=R0, Rcut=Rcut, zcut=zcut)
        elif flaw=='thick':
            if (zd is not None) or (ffit_array is not None):
                return Gaussian_disc.thick(sigma0=sigma0,sigmad=sigmad, R0=R0, zd=zd, zlaw=zlaw, ffit_array=ffit_array, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('zd or ffit_array must be a non None value for thick flaw')
        elif flaw=='poly':
            if (polycoeff is not None) or (ffit_array is not None):
                if fitdegree is None: fitdegree=3
                return Gaussian_disc.polyflare(sigma0=sigma0,sigmad=sigmad, R0=R0, polycoeff=polycoeff, zlaw=zlaw, ffit_array=ffit_array, fitdegree=fitdegree, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('polycoeff or ffit_array must be a non None value for poly flaw')
        elif flaw=='asinh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Gaussian_disc.asinhflare(sigma0=sigma0,sigmad=sigmad, R0=R0, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='tanh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Gaussian_disc.tanhflare(sigma0=sigma0,sigmad=sigmad, R0=R0, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')
        else:
            raise ValueError('Flaw %s does not exist: chose between thin, thick, poly, asinh, tanh'%flaw)

    def take_radial_from(self, cls):

        if isinstance(cls, PolyExponential_disc) == False:
            raise ('Value Error: dynamic component mismatch, given %s required %s' % (cls.name, self.name))

        self.sigma0 = cls.sigma0
        self.rparam = cls.rparam
        self.rlaw = cls.rlaw
        self.R0 = cls.R0
        self.sigmad= cls.sigmad
        self.Rcut = cls.Rcut


    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2e Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='sigmad: %.3f kpc \n'%self.sigmad
        s+='R0: %.3f kpc \n'%self.R0
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.fparam)
        s+='Rcut: %.3f kpc \n'%self.Rcut
        s+='zcut: %.3f kpc \n'%self.zcut
        if self.Rlimit is None:
            s+='Rlimit: None \n'
        else:
            s+='Rlimit: %.3f kpc \n'%self.Rlimit
        return s

class Frat_disc(disc):

    def __init__(self,sigma0,Rd, alpha, fparam, Rd2=None, zlaw='gau', flaw='poly',Rcut=50, zcut=30):


        self.sigma0=sigma0
        self.Rd=Rd
        if Rd2 is None: self.Rd2=Rd
        else: self.Rd2=Rd2
        self.alpha=alpha
        self.zlaw=zlaw
        self.Rcut=50
        self.zcut=30

        rparam=np.zeros(10)
        rparam[0] = Rd
        rparam[1] = Rd2
        rparam[2] = alpha

        super(Frat_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='fratlaw',flaw=flaw,Rcut=Rcut, zcut=zcut)
        self.name='Frat disc'


    @classmethod
    def thin(cls, sigma0=None, Rd=None, Rd2=None, alpha=None, rfit_array=None,Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_fratlaw(rfit_array,nproc=nproc)
            sigma0, Rd, Rd2, alpha = popt
        elif (sigma0 is not None) and (Rd is not None) and (alpha is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        fparam = np.array([0.0, 0])

        return cls(sigma0=sigma0,Rd=Rd, alpha=alpha, Rd2=Rd2, fparam=fparam,zlaw='dirac',flaw='constant', Rcut=Rcut, zcut=zcut)


    @classmethod
    def thick(cls,sigma0=None, Rd=None, Rd2=None, alpha=None, zd=None, rfit_array=None, ffit_array=None, zlaw='gau',Rcut=50, zcut=30,check_thin=True,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_fratlaw(rfit_array,nproc=nproc)
            sigma0, Rd, Rd2, alpha = popt
        elif (sigma0 is not None) and (Rd is not None)  and (alpha is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit=lambda R,zd: np.where(R==0,zd,zd)
            p0=(np.median(ffit_array[:,1]),)
            popt,pcov=_fit_utility(func_fit,ffit_array,p0)
            zd=popt[0]
        elif (zd is not None):
            pass
        else:
            raise ValueError()

        if check_thin:
            if zd<0.01:
                print('Warning Zd lower than 0.01, switching to thin disc')
                fparam=np.array([0,0])
                zlaw='dirac'
            else:
                fparam=np.array([zd,0])
        else:
            fparam = np.array([zd, 0])

        return cls(sigma0=sigma0, Rd=Rd, alpha=alpha, Rd2=Rd2, fparam=fparam, zlaw=zlaw, flaw='constant', Rcut=Rcut, zcut=zcut)

    @classmethod
    def polyflare(cls,sigma0=None, Rd=None, Rd2=None, alpha=None, polycoeff=None, zlaw='gau', rfit_array=None, ffit_array=None, fitdegree=4, Rlimit=None,Rcut=50, zcut=30,**kwargs):


        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_fratlaw(rfit_array,nproc=nproc)
            sigma0, Rd, Rd2, alpha = popt
        elif (sigma0 is not None) and (Rd is not None)  and (alpha is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            if fitdegree>7:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)' % lenp)
            else:
                popt,pcov=_fit_utility_poly(fitdegree, ffit_array)
                polycoeff=popt
                lenp = len(polycoeff)
        elif polycoeff is not None:
            lenp=len(polycoeff)
            if lenp>8:
                raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)
        else:
            raise ValueError()

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        cls_ret=cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='poly', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    @classmethod
    def asinhflare(cls, sigma0=None, Rd=None, Rd2=None, alpha=None, h0=None, Rf=None, c=None, zlaw=None, rfit_array=None, ffit_array=None, Rlimit=None,Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_fratlaw(rfit_array,nproc=nproc)
            sigma0, Rd, Rd2, alpha = popt
        elif (sigma0 is not None) and (Rd is not None)  and (alpha is not None):
            pass
        else:
            raise ValueError()


        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.arcsinh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.arcsinh(Rlimit * Rlimit/ (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='asinh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret


    @classmethod
    def tanhflare(cls, sigma0=None, Rd=None, Rd2=None, alpha=None, h0=None, Rf=None, c=None, zlaw='gau', rfit_array=None, ffit_array=None, Rlimit=None,Rcut=50, zcut=30,**kwargs):

        #Sigma(R)
        if rfit_array is not None:
            print('Fittin surface density profile...',end='')
            sys.stdout.flush()
            if 'nproc' in kwargs: nproc=kwargs['nproc']
            else: nproc=1
            popt,pcov,_=_fit_utility_fratlaw(rfit_array,nproc=nproc)
            sigma0, Rd, Rd2, alpha = popt
        elif (sigma0 is not None) and (Rd is not None)  and (alpha is not None):
            pass
        else:
            raise ValueError()

        #Flaw
        if ffit_array is not None:
            func_fit = lambda R, h0,Rf,c: h0+c*np.tanh(R*R/(Rf*Rf))
            p0 = (ffit_array[0, 1], 1, np.mean(ffit_array[:, 0]))
            popt, pcov = _fit_utility(func_fit, ffit_array, p0)
            h0,Rf,c = popt
        elif (h0 is not None) and (c is not None) and (Rf is not None):
            pass
        else:
            raise ValueError()


        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit/ (Rf*Rf))
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        cls_ret=cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='tanh', Rcut=Rcut, zcut=zcut)
        cls_ret.Rlimit=Rlimit

        return cls_ret

    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,ffit_array=None,fitdegree=None,Rlimit=None, zcut=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        Rd=self.Rd
        Rd2=self.Rd2
        alpha=self.alpha
        Rcut=self.Rcut
        if zcut is None:
            zcut=self.zcut
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return Frat_disc.thin(sigma0=sigma0,Rd=Rd,Rd2=Rd2,alpha=alpha, Rcut=Rcut, zcut=zcut)
        elif flaw=='thick':
            if (zd is not None) or (ffit_array is not None):
                return Frat_disc.thick(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, zd=zd,zlaw=zlaw, ffit_array=ffit_array, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('zd or ffit_array must be a non None value for thick flaw')
        elif flaw=='poly':
            if (polycoeff is not None) or (ffit_array is not None):
                if fitdegree is None: fitdegree=3
                return Frat_disc.polyflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, polycoeff=polycoeff,zlaw=zlaw, ffit_array=ffit_array, fitdegree=fitdegree, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('polycoeff or ffit_array must be a non None value for poly flaw')
        elif flaw=='asinh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Frat_disc.asinhflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='tanh':
            if ((h0 is not None) and (c is not None) and (Rf is not None) ) or (ffit_array is not None):
                return Frat_disc.tanhflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, h0=h0, c=c, Rf=Rf, zlaw=zlaw, ffit_array=ffit_array, Rlimit=Rlimit, Rcut=Rcut, zcut=zcut)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')
        else:
            raise ValueError('Flaw %s does not exist: chose between thin, thick, poly, asinh, tanh'%flaw)

    def take_radial_from(self, cls):

        if isinstance(cls, Frat_disc) == False:
            raise ('Value Error: dynamic component mismatch, given %s required %s' % (cls.name, self.name))

        self.sigma0 = cls.sigma0
        self.rparam = cls.rparam
        self.rlaw = cls.rlaw
        self.Rd = cls.Rd
        self.Rd2 = cls.Rd2
        self.alpha = cls.alpha
        self.Rcut = cls.Rcut

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2e Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.2f kpc\n'%self.Rd
        s+='Rd2: %.2f kpc\n'%self.Rd2
        s+='alpha: %.2f \n'%self.alpha
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n'%tuple(self.fparam)
        s+='Rcut: %.3f kpc \n'%self.Rcut
        s+='zcut: %.3f kpc \n'%self.zcut
        if self.Rlimit is None:
            s+='Rlimit: None \n'
        else:
            s+='Rlimit: %.3f kpc \n'%self.Rlimit
        return s