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
from .pot_disc import disc, _fit_utility, _fit_utility_poly


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