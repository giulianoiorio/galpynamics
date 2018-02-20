#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
import numpy as np
cimport numpy as np

cdef double PI=3.14159265358979323846

cdef double psi_iso(double d0, double rc, double m) nogil:
    """Auxiliary functions linked to density law iso:
    d=d0/(1+m/rc^2)

    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
    :param m:  elliptical radius
    :return:
    """

    return d0*rc*rc*(log(1+((m*m)/(rc*rc))))

cdef double integrand_hiso(int n, double *data) nogil:
    """ Potential integrand for isothermal halo: d=d0/(1+m/rc^2)

    :param n: dummy integer variable
    :param data: pointer to an array with
        0-m: integration variable (elliptical radius)
        1-R: Cylindrical radius
        2-Z: Cylindrical height
        3-mcut: elliptical radius where dens(m>mcut)=0
        4-d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
        5-rc: Core radius [Kpc]
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
        double rc = data[5]
        double e = data[6]
        double psi, result #, num, den

    if (m<=mcut): psi=psi_iso(d0,rc,m)
    else: psi=psi_iso(d0,rc,mcut)

    result=integrand_core(m, R, Z, e, psi)
    #num=xi(m,R,Z,e)*(xi(m,R,Z,e)-e*e)*sqrt(xi(m,R,Z,e)-e*e)*m*psi
    #den=((xi(m,R,Z,e)-e*e)*(xi(m,R,Z,e)-e*e)*R*R)+(xi(m,R,Z,e)*xi(m,R,Z,e)*Z*Z)

    return result


cdef double  _potential_iso(double R, double Z, double mcut, double d0, double rc, double e, double toll):
    """Calculate the potential of an isothermal halo in the point R-Z.
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius
    :param Z: Cylindrical height
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
    :param e: ellipticity
    :param toll: relative tollerance for the integration (see scipy.quad)
    :return: Potential of the isothermal halo in the point (R,Z)
    """

    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double intpot
        double result

    m0=m_calc(R,Z,e)

    #Integ
    import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hiso')

    intpot=quad(fintegrand,0.,m0,args=(R,Z,mcut,d0,rc,e),epsabs=toll,epsrel=toll)[0]


    psi=psi_iso(d0,rc,mcut)

    result=potential_core(e, intpot, psi)

    return result


cdef double[:,:]  _potential_iso_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rc, double e, double toll):
    """Calculate the potential of an isothermal halo in the a list of points R-Z
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param nlen: lentgh of R and Z
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
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
    import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hiso')

    for  i in range(nlen):


        ret[i,0]=R[i]
        ret[i,1]=Z[i]

        m0=m_calc(R[i],Z[i],e)

        intpot=quad(fintegrand,0.,m0,args=(R[i],Z[i],mcut,d0,rc,e),epsabs=toll,epsrel=toll)[0]

        psi=psi_iso(d0,rc,mcut)

        ret[i,2]=potential_core(e, intpot, psi)


        #if (e<=0.0001):
        #    ret[i,2] = -cost*(psi-intpot)
        #else:
        #    ret[i,2] = -cost*(sqrt(1-e*e)/e)*(psi*asin(e)-e*intpot)

    return ret

cdef double[:,:]  _potential_iso_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rc, double e, double toll):
    """Calculate the potential of an isothermal halo in a 2D grid combining the vector R and Z
        Use the formula 2.88b in BT 1987. The integration is performed with the function quad in scipy.quad.

    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param nlenR: lentgh of R
    :param nlenZ: lentgh of Z
    :param mcut: elliptical radius where dens(m>mcut)=0
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
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
    import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_hiso')

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]

            m0=m_calc(R[i],Z[j],e)

            intpot=quad(fintegrand,0.,m0,args=(R[i],Z[j],mcut,d0,rc,e),epsabs=toll,epsrel=toll)[0]

            psi=psi_iso(d0,rc,mcut)

            ret[c,2]=potential_core(e, intpot, psi)
            #if (e<=0.0001):
            #    ret[c,2] = -cost*(psi-intpot)
            #else:
            #    ret[c,2] = -cost*(sqrt(1-e*e)/e)*(psi*asin(e)-e*intpot)

            c+=1

    return ret


cpdef potential_iso(R, Z, d0, rc, e, mcut, toll=1e-4, grid=False):
    """Calculate the potential of an isothermal halo.
        If len(R)|=len(Z) or grid=True, calculate the potential in a 2D grid in R and Z.


    :param R: Cylindrical radius (memview object)
    :param Z: Cylindrical height (memview object)
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
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

    #print('R',R)
    #print('Z',Z)
    #print('d0',d0)
    #print('rc',rc)
    #print('e',e)
    #print('mcut',mcut)
    #print('toll',toll)
    #print('grid',grid)


    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            return np.array(_potential_iso(R=R,Z=Z,mcut=mcut,d0=d0,rc=rc,e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_iso_grid( R=R, Z=Z, nlenR=len(R), nlenZ=len(Z), mcut=mcut, d0=d0, rc=rc, e=e, toll=toll))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_iso_array( R=R, Z=Z, nlen=len(R), mcut=mcut, d0=d0, rc=rc, e=e, toll=toll))
        else:
            raise ValueError('R and Z have different dimension')

#####################################################################
#Vcirc
cdef double vcirc_integrand_iso(int n, double *data) nogil:
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
    dens=1/(1+x*x)

    return core*dens


cpdef  _vcirc_iso_spherical(R, d0, rc):
    """
    Circular velocity for a spherical halo (d=d0/(1+(r/rc)^2))
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
    :return: an array with the vcirc in km/s
    """
    cdef:
        double G=4.302113488372941e-06 #kpc km2/(msol s^2)

    x=R/rc
    vinf2=4*np.pi*G*d0*rc*rc
    vcirc=np.sqrt(vinf2*(1-np.arctan(x)/x))

    if isinstance(R, float) or isinstance(R, int):

        ret=vcirc

    else:

        ret=np.zeros((len(R),2))

        ret[:,0]=R
        ret[:,1]=vcirc

    return ret

cdef double _vcirc_iso(double R, double d0, double rc, double e, double toll):
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

    #Integ
    import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_iso')

    intvcirc=quad(fintegrand,0.,R,args=(R,rc,e),epsabs=toll,epsrel=toll)[0]
	
    result=sqrt(norm*intvcirc)

    return result


cdef double[:,:] _vcirc_iso_array(double[:] R, int nlen, double d0, double rc, double e, double toll):
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




    #Integ
    import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_iso')

    for  i in range(nlen):

        ret[i,0]=R[i]
        intvcirc=quad(fintegrand,0.,R[i],args=(R[i],rc,e),epsabs=toll,epsrel=toll)[0]
        ret[i,1]=sqrt(norm*intvcirc)

    return ret


cpdef vcirc_iso(R, d0, rc, e, toll=1e-4):
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

    if e==0:
        ret= _vcirc_iso_spherical(R, d0, rc)
    else:

        if isinstance(R, float) or isinstance(R, int):

            if R==0: ret=0
            else: ret= _vcirc_iso(R, d0, rc,  e,  toll)

        else:

            ret=_vcirc_iso_array(R, len(R), d0, rc, e, toll)
            ret[:,1]=np.where(R==0, 0, ret[:,1])

    return ret

