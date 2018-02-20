#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, pow, exp
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
import numpy as np
cimport numpy as np

cdef double PI=3.14159265358979323846


cdef double psi_exponential(double d0, double rb, double m) nogil:


    cdef:
        double costn=2*rb*rb
        double x=m/rb



    return d0*costn*(1-exp(-x)*(1+x))



cdef double integrand_exponential(int n, double *data) nogil:


    cdef:
        double m = data[0]

    if m==0.: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0

    cdef:
        double R = data[1]
        double Z = data[2]
        double mcut = data[3]
        double d0 = data[4]
        double rb = data[5]
        double e = data[6]
        double psi, result #, num, den

    if (m<=mcut): psi=psi_exponential(d0,rb,m)
    else: psi=psi_exponential(d0,rb,mcut)

    result=integrand_core(m, R, Z, e, psi)
    #num=xi(m,R,Z,e)*(xi(m,R,Z,e)-e*e)*sqrt(xi(m,R,Z,e)-e*e)*m*psi
    #den=((xi(m,R,Z,e)-e*e)*(xi(m,R,Z,e)-e*e)*R*R)+(xi(m,R,Z,e)*xi(m,R,Z,e)*Z*Z)

    return result


cdef double  _potential_exponential(double R, double Z, double mcut, double d0, double rb, double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double intpot
        double result

    m0=m_calc(R,Z,e)

    #Integ
    import discH.src.pot_halo.pot_c_ext.exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_exponential')

    intpot=quad(fintegrand,0.,m0,args=(R,Z,mcut,d0,rb,e),epsabs=toll,epsrel=toll)[0]


    psi=psi_exponential(d0,rb,mcut)

    result=potential_core(e, intpot, psi)

    return result


cdef double[:,:]  _potential_exponential_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rb, double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        double intpot
        int i



    #Integ
    import discH.src.pot_halo.pot_c_ext.exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_exponential')


    for  i in range(nlen):


        ret[i,0]=R[i]
        ret[i,1]=Z[i]

        m0=m_calc(R[i],Z[i],e)

        intpot=quad(fintegrand,0.,m0,args=(R[i],Z[i],mcut,d0,rb,e),epsabs=toll,epsrel=toll)[0]

        psi=psi_exponential(d0,rb,mcut)

        ret[i,2]=potential_core(e, intpot, psi)


    return ret



cdef double[:,:]  _potential_exponential_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rb, double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_halo.pot_c_ext.exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_exponential')

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]

            m0=m_calc(R[i],Z[j],e)

            intpot=quad(fintegrand,0.,m0,args=(R[i],Z[j],mcut,d0,rb,e),epsabs=toll,epsrel=toll)[0]

            psi=psi_exponential(d0,rb,mcut)

            ret[c,2]=potential_core(e, intpot, psi)

            c+=1

    return ret


cpdef potential_exponential(R, Z, d0, rb, e, mcut, toll=1e-4, grid=False):

    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            return np.array(_potential_exponential(R=R,Z=Z,mcut=mcut,d0=d0,rb=rb,e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_exponential_grid( R=R, Z=Z, nlenR=len(R), nlenZ=len(Z), mcut=mcut, d0=d0, rb=rb, e=e, toll=toll))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_exponential_array( R=R, Z=Z, nlen=len(R), mcut=mcut, d0=d0, rb=rb, e=e, toll=toll))
        else:
            raise ValueError('R and Z have different dimension')

#####################################################################
#Vcirc
cdef double vcirc_integrand_exponential(int n, double *data) nogil:
    """
    Integrand function for vcirc  on the plane (Eq. 2.132 in BT2)
    :param m:
    :param R:
    :param rb:
    :param e:
    :return:
    """


    cdef:
        double m = data[0]
        double R = data[1]
        double rb = data[2]
        double e = data[3]
        double core
        double dens
        double x=m/rb
        double base
    core=vcirc_core(m, R, e)
    dens=exp(-x)

    return core*dens

cdef double _vcirc_exponential(double R, double d0, double rb, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rb: Scale radius (kpc)
    :param e: ellipticity
    :return:
    """

    cdef:
        double G=4.302113488372941e-06 #G constant in  kpc km2/(msol s^2)
        double cost=4*PI*G
        double norm=cost*sqrt(1-e*e)*d0
        double intvcirc
        double result

    #Integ
    import discH.src.pot_halo.pot_c_ext.exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_exponential')

    intvcirc=quad(fintegrand,0.,R,args=(R,rb,e),epsabs=toll,epsrel=toll)[0]

    result=sqrt(norm*intvcirc)

    return result


cdef double[:,:] _vcirc_exponential_array(double[:] R, int nlen, double d0, double rb, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rb: Scale radius (kpc)
    :param e: ellipticity
    :return:
    """

    cdef:
        double G=4.302113488372941e-06 #G constant in  kpc km2/(msol s^2)
        double cost=4*PI*G
        double norm=cost*sqrt(1-e*e)*d0
        double intvcirc
        int i
        double[:,:] ret=np.empty((nlen,2), dtype=np.dtype("d"))




    #Integ
    import discH.src.pot_halo.pot_c_ext.exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_exponential')

    for  i in range(nlen):

        ret[i,0]=R[i]
        intvcirc=quad(fintegrand,0.,R[i],args=(R[i],rb,e),epsabs=toll,epsrel=toll)[0]
        ret[i,1]=sqrt(norm*intvcirc)

    return ret

cpdef vcirc_exponential(R, d0, rb, e, toll=1e-4):
    """Calculate the Vcirc on the plane of an isothermal halo.

    :param R: Cylindrical radius (memview object)
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rb: Scale radius [Kpc]
    :param e: ellipticity
    :param e: ellipticity
    :param toll: Tollerance for nquad
    :return: 2-col array:
        0-R
        1-Vcirc(R)
    """



    if isinstance(R, float) or isinstance(R, int):

        if R==0: ret=0
        else: ret= _vcirc_exponential(R, d0, rb,  e,  toll)

    else:

        ret=_vcirc_exponential_array(R, len(R), d0, rb, e, toll)
        ret[:,1]=np.where(R==0, 0, ret[:,1])

    return ret
