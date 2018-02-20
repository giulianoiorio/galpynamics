#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
import numpy as np
cimport numpy as np


cdef double PI=3.14159265358979323846

cdef double psi_nfw(double d0,double rs,double m) nogil:
    """Auxiliary functions linked to density law nfw:
    d=d0/((m/rs)*(1+m/rs)^2)

    :param d0: Typical density at m/rc=0.465571... (nfw diverges in the center)
    :param rs: Scale radius [Kpc]
    :param m: spheroidal radius
    :return:
    """

    cdef:
        double val=(1/(1+m/rs))-1

    return -2*d0*rs*rs*val
    #cdef:
    #    double num= 2*m*rs*rs
    #    double den= m+rs

    #return d0*num/den



cdef double integrand_hnfw(int n, double *data) nogil:
    """ Potential integrand for nfw halo: d=d0/((m/rs)*(1+m/rs)^2)

    :param n: dummy integer variable
    :param data: pointer to an array with
        0-m: integration variable (elliptical radius)
        1-R: Cylindrical radius
        2-Z: Cylindrical height
        3-mcut: elliptical radius where dens(m>mcut)=0
        4-d0: Typical density at m/rs=0.465571... (nfw diverges in the center) [Msol/kpc^3]
        5-rs: Core radius [Kpc]
        6-e: ellipticity
    :return: integrand function
    """

    cdef:
        double m = data[0]

    if m==0.: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0

    cdef:
        double R = data[1]
        double Z = data[2]
        double mcut = data[3]
        double d0 = data[4]
        double rs = data[5]
        double e = data[6]
        double psi, result #, num, den

    if (m<=mcut): psi=psi_nfw(d0,rs,m)
    else: psi=psi_nfw(d0,rs,mcut)

    result=integrand_core(m, R, Z, e, psi)

    return result

cdef double _potential_nfw(double R, double Z, double mcut, double d0, double rs, double e, double toll):
    """Calculate the potential of a NFW halo in the point R-Z.
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius
    :param Z: Cylindrical height
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Typical density at m/rs=0.465571... (nfw diverges in the center) [Msol/kpc^3]
    :param rc: Scale radius [Kpc]
    :param e: ellipticity
    :param toll: relative tollerance for the integration (see scipy.quad)
    :return: Potential of the nfw halo in the point (R,Z)
    """

    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double intpot
        double result

    m0=m_calc(R,Z,e)

    import discH.src.pot_halo.pot_c_ext.nfw_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hnfw')

    intpot=quad(fintegrand,0.,m0,args=(R,Z,mcut,d0,rs,e),epsabs=toll,epsrel=toll)[0]

    psi=psi_nfw(d0,rs,mcut)

    result=potential_core(e, intpot, psi)

    return result

cdef double[:,:]  _potential_nfw_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double e, double toll):
    """Calculate the potential of an isothermal halo in the a list of points R-Z
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param nlen: lentgh of R and Z
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Typical density at m/rs=0.465571... (nfw diverges in the center) [Msol/kpc^3]
    :param rs: Scale radius [Kpc]
    :param e: ellipticity
    :param toll: relative tollerance for the integration (see scipy.quad)
    :return: 3-col array:
        0-R
        1-Z
        2-Potential at (R,Z)
    """

    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        double intpot
        int i



    #Integ
    import discH.src.pot_halo.pot_c_ext.nfw_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hnfw')

    for  i in range(nlen):


        ret[i,0]=R[i]
        ret[i,1]=Z[i]

        m0=m_calc(R[i],Z[i],e)

        intpot=quad(fintegrand,0.,m0,args=(R[i],Z[i],mcut,d0,rs,e),epsabs=toll,epsrel=toll)[0]

        psi=psi_nfw(d0,rs,mcut)

        ret[i,2]=potential_core(e, intpot, psi)



    return ret

cdef double[:,:]  _potential_nfw_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double e, double toll):
    """Calculate the potential of a NFW halo in a 2D grid combining the vector R and Z
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param nlenR: lentgh of R
    :param nlenZ: lentgh of Z
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Typical density at m/rs=0.465571... (nfw diverges in the center) [Msol/kpc^3]
    :param rs: Scale radius [Kpc]
    :param e: ellipticity
    :param toll: relative tollerance for the integration (see scipy.quad)
    :return: 3-col array:
        0-R
        1-Z
        2-Potential at (R,Z)
    """

    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_halo.pot_c_ext.nfw_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hnfw')

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]

            m0=m_calc(R[i],Z[j],e)

            intpot=quad(fintegrand,0.,m0,args=(R[i],Z[j],mcut,d0,rs,e),epsabs=toll,epsrel=toll)[0]

            psi=psi_nfw(d0,rs,mcut)

            ret[c,2]=potential_core(e, intpot, psi)


            c+=1

    return ret

cpdef potential_nfw(R, Z, d0, rs, e, mcut, toll=1e-4, grid=False):
    """Calculate the potential of a NFW halo.
        If len(R)|=len(Z) or grid=True, calculate the potential in a 2D grid in R and Z.


    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param d0: Typical density at m/rs=0.465571... (nfw diverges in the center) [Msol/kpc^3]
    :param rs: scale radius [Kpc]
    :param e: ellipticity
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param e: ellipticity
    :param toll: Tollerance for nquad
    :param grid: If True calculate potential in a 2D grid in R and Z
    :return: 3-col array:
        0-R
        1-Z
        2-Potential at (R,Z)
    """



    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            return np.array(_potential_nfw(R=R,Z=Z,mcut=mcut,d0=d0,rs=rs,e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_nfw_grid( R=R, Z=Z, nlenR=len(R), nlenZ=len(Z), mcut=mcut, d0=d0, rs=rs, e=e, toll=toll))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_nfw_array( R=R, Z=Z, nlen=len(R), mcut=mcut, d0=d0, rs=rs, e=e, toll=toll))
        else:
            raise ValueError('R and Z have different dimension')





#####################################################################
#Vcirc
cdef double vcirc_integrand_nfw(int n, double *data) nogil:
    """
    Integrand function for vcirc  on the plane (Eq. 2.132 in BT2)
    :param m:
    :param R:
    :param rc:
    :param e:
    :return:
    """


    cdef:
        double m = data[0]
        double R = data[1]
        double rc = data[2]
        double e = data[3]
        double core
        double dens
        double x=m/rc

    core=vcirc_core(m, R, e)
    dens=1/((x)*(1+x)*(1+x))

    return core*dens


cpdef  _vcirc_nfw_spherical(R, d0, rs):
    """
    Circular velocity for a spherical halo (d=d0/(1+(r/rc)^2))
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
    :return: an array with the vcirc in km/s
    """
    cdef:
        double G=4.302113488372941e-06 #kpc km2/(msol s^2)
        double rc=rs

    x=R/rc
    cost=4*np.pi*G*d0*rs*rs
    vcirc=np.sqrt((cost/x)*(np.log(1+x) - (x)/(1+x)))

    if isinstance(R, float) or isinstance(R, int):

        ret=vcirc

    else:

        ret=np.zeros((len(R),2))

        ret[:,0]=R
        ret[:,1]=vcirc

    return ret

cdef double _vcirc_nfw(double R, double d0, double rs, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
    :param e: ellipticity
    :return:
    """

    cdef:
        double G=4.302113488372941e-06 #G constant in  kpc km2/(msol s^2)
        double cost=4*PI*G
        double norm=cost*sqrt(1-e*e)*d0
        double intvcirc
        double result
        double rc=rs

    #Integ
    import discH.src.pot_halo.pot_c_ext.nfw_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_nfw')

    intvcirc=quad(fintegrand,0.,R,args=(R,rc,e),epsabs=toll,epsrel=toll)[0]

    result=sqrt(norm*intvcirc)

    return result


cdef double[:,:] _vcirc_nfw_array(double[:] R, int nlen, double d0, double rs, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
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
        double rc=rs



    #Integ
    import discH.src.pot_halo.pot_c_ext.nfw_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_nfw')

    for  i in range(nlen):

        ret[i,0]=R[i]
        intvcirc=quad(fintegrand,0.,R[i],args=(R[i],rc,e),epsabs=toll,epsrel=toll)[0]
        ret[i,1]=sqrt(norm*intvcirc)

    return ret


cpdef vcirc_nfw(R, d0, rs, e, toll=1e-4):
    """Calculate the Vcirc on the plane of an isothermal halo.

    :param R: Cylindrical radius (memview object)
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
    :param e: ellipticity
    :param e: ellipticity
    :param toll: Tollerance for nquad
    :return: 2-col array:
        0-R
        1-Vcirc(R)
    """

    rc=rs

    if e==0:
        ret= _vcirc_nfw_spherical(R, d0, rc)
    else:

        if isinstance(R, float) or isinstance(R, int):

            if R==0: ret=0
            else: ret= _vcirc_nfw(R, d0, rc,  e,  toll)

        else:

            ret=_vcirc_nfw_array(R, len(R), d0, rc, e, toll)
            ret[:,1]=np.where(R==0, 0, ret[:,1])

    return ret