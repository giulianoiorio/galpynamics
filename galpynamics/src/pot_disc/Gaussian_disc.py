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