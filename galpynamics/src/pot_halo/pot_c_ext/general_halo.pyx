#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, asin

cdef double PI=3.14159265358979323846


cdef double xi(double m,double R,double Z, double e) nogil:
    """Aux function for halo integration

    :param m:  elliptical radius
    :param R:  Cylindrical radius
    :param Z:  Cylindrical height
    :param e:  eccentricity
    :return:   xi function
    """

    return  (R*R+Z*Z+e*e*m*m+sqrt((e*m)**4-2*e*e*m*m*(R*R-Z*Z)+(R*R+Z*Z)*(R*R+Z*Z)))/(2*m*m)

cdef double m_calc(double R, double Z, double e) nogil:
    """Calculate the elliptical radius

    :param R: Cylindrical radius
    :param Z: Cylindrical height
    :param e: eccentricity
    :return:  elliptical radius
    """

    cdef:
        double q2=(1-e*e)

    return sqrt(R*R+Z*Z/q2)



cdef double integrand_core(double m, double R, double Z, double e, double psi) nogil:
    """
    Integrand core function for halo potential
    :param m: Elliptical Radius
    :param R: Cylindrical Radial coordinate
    :param Z: Cylindrical Vertical coordinate
    :param e: eccentricity
    :param psi: Psi value, depends on the halo model
    :return:
    """

    cdef:
        double num, den, xieval

    xieval=xi(m,R,Z,e)

    num=xieval*(xieval-e*e)*sqrt(xieval-e*e)*m*psi
    den=((xieval-e*e)*(xieval-e*e)*R*R)+(xieval*xieval*Z*Z)

    return num/den


cdef double potential_core(double e, double intpot, double psi) nogil:
    """Function to calcualte the potential (Use the formula 2.88b in BT 1987)

    :param e: ellipticity
    :param intpot: result of the integration
    :param psi:  Psi value, depends on the halo model
    :return:  Final potential
    """

    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G

    if (e<=0.0001): return -cost*(psi-intpot)
    else: return -cost*(sqrt(1-e*e)/e)*(psi*asin(e)-e*intpot)

cdef double vcirc_core(double m, double R, double e) nogil:
    """
    Core function to calculate the Vcirc of a flattened ellipsoids (Eq. 2.132 BT2)
    :param m: integrand variable
    :param R: Radius on the meridional plane
    :param e: flattening
    :return:
    """

    cdef:
        double m2=m*m
        double den

    den=sqrt(R*R - m2*e*e)

    return m2/den


cpdef sph_to_ell(hlaw,d0,rc,e):
    """
    Find the value of d0 and rc for an halo of ellipticity e, starting from the values d0 and rc for a spherical halo.
    The transformation is performed to have the same rotation curve. (App. C, my thesis)
    :param hlaw: 'iso' for isothermal, 'nfw' for navarro-frank-white
    :param d0:
    :param rc:
    :param e:
    :return: d0(e), rc(e)
    """
    q=sqrt(1-e*e)
    if hlaw=='iso':
        kappa=0.598+(0.996/(q+1.460))-(0.003*q*q*q)
        lamb=0.538+(0.380/q)+0.083*q
    elif hlaw=='nfw':
        kappa=0.549+(1.170/(q+1.367))-0.047*q
        lamb=0.510+(0.312/q)+0.178*q
    else: raise IOError('hlaw allowed: iso or nfw')
    d0n=d0*lamb
    rcn=rc*kappa
    return d0n,rcn