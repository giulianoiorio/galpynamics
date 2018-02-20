#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, fabs, cosh
from cython_gsl cimport *
from .rdens_law cimport poly_exponential, gaussian, fratlaw, poly_exponential_der, gaussian_der, fratlaw_der
from .rflare_law cimport poly_flare, constant, tanh_flare, asinh_flare, poly_flare_der, constant_der, tanh_flare_der, asinh_flare_der
from .zdens_law cimport zexp, zexp_der, zgau, zgau_der, zsech2, zsech2_der
from scipy._lib._ccallback import LowLevelCallable
from scipy.integrate import nquad, quad
cimport numpy as np
import numpy as np
import ctypes
cdef double PI=3.14159265358979323846
from .model_option import checkrd_dict, checkfl_dict



######
#ZEXP
cdef double rhoder_zexp(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Exp(-l/zd)/(2zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The radial derivative of the 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz, densr_der, densz_der, zd_der, rhoder
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1:
        densr      = poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = poly_exponential_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==2:
        densr      = fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = fratlaw_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==3:
        densr      = gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = gaussian_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    #Flare law
    if checkfli==0 :
        zd         = constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = 0.

    elif checkfli==1:
        zd         = poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = poly_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==2:
        zd         = asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = asinh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==3:
        zd         = tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = tanh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    densz       =   zexp(l,zd)
    densz_der   =   zexp_der(l,zd)

    rhoder      =   densr_der*densz     +   densr*densz_der*zd_der



    return rhoder

#Vcirc calc
cdef double integrand_vcirc_zexp(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Exp(-l/zd)/(2zd)

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-checkrd, number to choice the radial surface density law
        5-checkfl, number to choice the radial flaring law
        6-15 rcoeff, Params of the radial surface density law
        16-26 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=25

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[3]
        double checkfl = data[4]
        double d0 = data[5] #Rd
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double d6 = data[11]
        double d7 = data[12]
        double d8 = data[13]
        double d9 = data[14]
        double f0 = data[15]
        double f1 = data[16]
        double f2 = data[17]
        double f3 = data[18]
        double f4 = data[19]
        double f5 = data[20]
        double f6 = data[21]
        double f7 = data[22]
        double f8 = data[23]
        double f9 = data[24]
        double x, p, diff_ellint, factor, ellinte, ellintk, rho_der


    x               =   (R*R + u*u + l*l)/(2*R*u)

    if x==1:
        return 0

    p               =   x - sqrt(x*x-1)
    factor          =   sqrt(u/p)
    ellinte         =   gsl_sf_ellint_Ecomp(p, GSL_PREC_DOUBLE)
    ellintk         =   gsl_sf_ellint_Kcomp(p, GSL_PREC_DOUBLE)
    diff_ellint     =   ellintk - ellinte
    rho_der         =   rhoder_zexp(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)


    return factor * diff_ellint * rho_der


######
#ZGAU
cdef double rhoder_zgau(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The radial derivative of the 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz, densr_der, densz_der, zd_der, rhoder
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1:
        densr      = poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = poly_exponential_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==2:
        densr      = fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = fratlaw_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==3:
        densr      = gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = gaussian_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)


    #Flare law
    if checkfli==0 :
        zd         = constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = 0.

    elif checkfli==1:
        zd         = poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = poly_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==2:
        zd         = asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = asinh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==3:
        zd         = tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = tanh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    densz       =   zgau(l,zd)
    densz_der   =   zgau_der(l,zd)

    rhoder      =   densr_der*densz     +   densr*densz_der*zd_der

    #print('zd',zd)
    #print('zd_der', zd_der)
    #print('densr_der',densr_der)
    #print('densr',densr)
    #print('dz',densz)
    #print('dz_der',densz_der)
    #print('rhoder',rhoder)

    return rhoder

#Vcirc calc
cdef double integrand_vcirc_zgau(int n, double *data) nogil:
    """Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    l and zd need to have the same physical units

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-checkrd, number to choice the radial surface density law
        5-checkfl, number to choice the radial flaring law
        6-15 rcoeff, Params of the radial surface density law
        16-26 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=25

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[3]
        double checkfl = data[4]
        double d0 = data[5] #Rd
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double d6 = data[11]
        double d7 = data[12]
        double d8 = data[13]
        double d9 = data[14]
        double f0 = data[15]
        double f1 = data[16]
        double f2 = data[17]
        double f3 = data[18]
        double f4 = data[19]
        double f5 = data[20]
        double f6 = data[21]
        double f7 = data[22]
        double f8 = data[23]
        double f9 = data[24]
        double x, p, diff_ellint, factor, ellinte, ellintk, rho_der


    x               =   (R*R + u*u + l*l)/(2*R*u)
    if x==1:
        return 0


    p               =   x - sqrt(x*x-1)
    factor          =   sqrt(u/p)
    ellinte         =   gsl_sf_ellint_Ecomp(p, GSL_PREC_DOUBLE)
    ellintk         =   gsl_sf_ellint_Kcomp(p, GSL_PREC_DOUBLE)
    diff_ellint     =   ellintk - ellinte
    rho_der         =   rhoder_zgau(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)


    return factor * diff_ellint * rho_der

######
#ZSECH2
cdef double rhoder_zsech2(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Sech(-l/zd)^2
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The radial derivative of the 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz, densr_der, densz_der, zd_der, rhoder
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1:
        densr      = poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = poly_exponential_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==2:
        densr      = fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = fratlaw_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    elif checkrdi==3:
        densr      = gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        densr_der  = gaussian_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    #Flare law
    if checkfli==0 :
        zd         = constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = 0.

    elif checkfli==1:
        zd         = poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = poly_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==2:
        zd         = asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = asinh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    elif checkfli==3:
        zd         = tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
        zd_der     = tanh_flare_der(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)

    densz       =   zsech2(l,zd)
    densz_der   =   zsech2_der(l,zd)

    rhoder      =   densr_der*densz     +   densr*densz_der*zd_der


    return rhoder

#Vcirc calc
cdef double integrand_vcirc_zsech2(int n, double *data) nogil:
    """Vertical law: Sech(-l/zd)^2
    l and zd need to have the same physical units

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-checkrd, number to choice the radial surface density law
        5-checkfl, number to choice the radial flaring law
        6-15 rcoeff, Params of the radial surface density law
        16-26 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=25

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[3]
        double checkfl = data[4]
        double d0 = data[5] #Rd
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double d6 = data[11]
        double d7 = data[12]
        double d8 = data[13]
        double d9 = data[14]
        double f0 = data[15]
        double f1 = data[16]
        double f2 = data[17]
        double f3 = data[18]
        double f4 = data[19]
        double f5 = data[20]
        double f6 = data[21]
        double f7 = data[22]
        double f8 = data[23]
        double f9 = data[24]
        double x, p, diff_ellint, factor, ellinte, ellintk, rho_der


    x               =   (R*R + u*u + l*l)/(2*R*u)
    if x==1:
        return 0
    p               =   x - sqrt(x*x-1)
    factor          =   sqrt(u/p)
    ellinte         =   gsl_sf_ellint_Ecomp(p, GSL_PREC_DOUBLE)
    ellintk         =   gsl_sf_ellint_Kcomp(p, GSL_PREC_DOUBLE)
    diff_ellint     =   ellintk - ellinte
    rho_der         =   rhoder_zsech2(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)


    return factor * diff_ellint * rho_der

#Delta dirac
######
cdef double rhoder_zdirac(double u,  double checkrd, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9) nogil:
    """Vertical law: delta(Z=0)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param checkrd: Option to choice the radial surface density law
    :param d0-d9: Params of the radial surface density law
    :return: The radial derivative of 2D density at R=u
    """

    cdef:
        double  norm, densr_der, rhoder
        int checkrdi=<int> checkrd

    #Dens law
    if   checkrdi==1: densr_der = poly_exponential_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==2: densr_der = fratlaw_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==3: densr_der = gaussian_der(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    rhoder  =   densr_der


    return rhoder

cdef double integrand_vcirc_zdirac(int n, double *data) nogil:
    """Integrand function for
    Vertical law: delta(Z=0)

    :param data:
        0-u, Radial integration variable
        1-R, Radial position
        2-Z, Vertical position
        3-checkrd, number to choice the radial surface density law
        4-13 rcoeff, Params of the radial surface density law
    :return: Value of the integrand function
    """

    n=13

    cdef:
        double u = data[0] #R intengration variable
        double R = data[1]


    if (R==0) | (u==0) | (R==u): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[2]
        double d0 = data[3]
        double d1 = data[4]
        double d2 = data[5]
        double d3 = data[6]
        double d4 = data[7]
        double d5 = data[8]
        double d6 = data[9]
        double d7 = data[10]
        double d8 = data[11]
        double d9 = data[12]
        double x, p, diff_ellint, factor, ellinte, ellintk, rho_der


    x               =   (R*R + u*u)/(2*R*u)
    p               =   x - sqrt(x*x-1)
    factor          =   sqrt(u/p)
    ellinte         =   gsl_sf_ellint_Ecomp(p, GSL_PREC_DOUBLE)
    ellintk         =   gsl_sf_ellint_Kcomp(p, GSL_PREC_DOUBLE)
    diff_ellint     =   ellintk - ellinte
    rho_der         =   rhoder_zdirac(u, checkrd, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)


    return factor * diff_ellint * rho_der

#######
#Vcirc THICK
cdef double _vcirc_disc(double R, int zlaw, double sigma0, double checkrd, double checkfl, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """Vcirc disc for a single value of R
    :param R: Radial coordinate  (float or int)
    :param zlaw: exp, gau or sech2
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param checkfl: number to choice the radial flaring law
    :param rparam:  Params of the radial surface density law
    :param fparam: Params of the radial flaring law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :param zcut: Vertical cut of the density
    :return: Potential at R and Z in kpc/Myr
    """

    cdef:
        double G=4.518359396265313e-39 #kpc^3/(msol s^2)
        double kpc_to_km=3.08567758e16 #kpc_to_km
        double cost=(-8*G*sigma0*sqrt(R))
        double intvcirc, vc2, vc



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_vcirc as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zexp')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zsech2')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zgau')


    cdef:
        double d0=rparam[0]
        double d1=rparam[1]
        double d2=rparam[2]
        double d3=rparam[3]
        double d4=rparam[4]
        double d5=rparam[5]
        double d6=rparam[6]
        double d7=rparam[7]
        double d8=rparam[8]
        double d9=rparam[9]
        double f0=fparam[0]
        double f1=fparam[1]
        double f2=fparam[2]
        double f3=fparam[3]
        double f4=fparam[4]
        double f5=fparam[5]
        double f6=fparam[6]
        double f7=fparam[7]
        double f8=fparam[8]
        double f9=fparam[9]


    intvcirc=nquad(fintegrand,[[0.,rcut],[0.,zcut]],args=(R,checkrd,checkfl,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[0,],'epsabs':toll,'epsrel':toll})])[0]


    vc2=cost*intvcirc

    if vc2>0:
        vc=sqrt(vc2) #vc in kpc/s
    else:
        vc=-sqrt(-vc2) #vc in kpc/s


    return vc*kpc_to_km #vc in km/s

cpdef double[:,:] _vcirc_disc_array(double[:] R, int nlen, int zlaw, double sigma0, double checkrd, double checkfl, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """Vcirc disc for an array of R
    :param R: Radial coordinate  (list)
    :param nlen: length of the array R and Z (they should have the same dimension)
    :param zlaw: exp, gau or sech2
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param checkfl: number to choice the radial flaring law
    :param rparam:  Params of the radial surface density law
    :param fparam: Params of the radial flaring law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :param zcut: Vertical cut of the density
    :return: Potential at R and Z in kpc/Myr
    """

    cdef:
        double G=4.518359396265313e-39 #kpc^3/(msol s^2)
        double kpc_to_km=3.08567758e16 #kpc_to_km
        double cost=-(8*G*sigma0)
        double intvcirc, vc2, vc, R_tmp
        double[:,:] ret=np.empty((nlen,2), dtype=np.dtype("d"))
        int i



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_vcirc as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zexp')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zsech2')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zgau')


    cdef:
        double d0=rparam[0]
        double d1=rparam[1]
        double d2=rparam[2]
        double d3=rparam[3]
        double d4=rparam[4]
        double d5=rparam[5]
        double d6=rparam[6]
        double d7=rparam[7]
        double d8=rparam[8]
        double d9=rparam[9]
        double f0=fparam[0]
        double f1=fparam[1]
        double f2=fparam[2]
        double f3=fparam[3]
        double f4=fparam[4]
        double f5=fparam[5]
        double f6=fparam[6]
        double f7=fparam[7]
        double f8=fparam[8]
        double f9=fparam[9]


    for  i in range(nlen):

        R_tmp=R[i]

        ret[i,0]=R_tmp
        intvcirc=nquad(fintegrand,[[0.,rcut],[0.,zcut]],args=(R_tmp,checkrd,checkfl,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9),opts=[({'points':[0,R_tmp],'epsabs':toll,'epsrel':toll}),({'points':[0,],'epsabs':toll,'epsrel':toll})])[0]

        vc2=cost*intvcirc*sqrt(R_tmp)

        if vc2>0:
            vc  =   sqrt(vc2) #vc in kpc/s
        else:
            vc  =   -sqrt(-vc2) #vc in kpc/s


        ret[i,1]=vc*kpc_to_km #vc in km/s

    return ret

cpdef vcirc_disc(R, sigma0, rcoeff, fcoeff, zlaw='gau', rlaw='epoly', flaw='poly', rcut=None, zcut=None, toll=1e-4):
    """Wrapper for the vcirc of a disc

    :param R: Radial coordinates
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param rcoeff: Params of the radial surface density law
    :param fcoeff: Params of the radial flaring law
    :param zlaw: Vertical density law (exp, gau or sech2)
    :param rlaw: Radial surface density law
    :param flaw: Radial flaring law
    :param rcut: Radial cut of the density
    :param zcut: Vertical cut of the density
    :param toll: Relative tollerance for quad and nquad
    :return:  Vcirc at R in km/s
    """


    if zlaw=='exp': izdens=0
    elif zlaw=='sech2': izdens=1
    elif zlaw=='gau': izdens=2
    else: raise NotImplementedError('Z-Dens law %s not implmented'%zlaw)

    #Flaw
    if flaw in checkfl_dict: checkfl=checkfl_dict[flaw]
    else: raise NotImplementedError('Flare law %s not implmented'%flaw)

    #Rdens
    if rlaw in checkrd_dict: checkrd=checkrd_dict[rlaw]
    else: raise NotImplementedError('Dens law %s not implmented'%rlaw)


    rparam=np.array(rcoeff,dtype=np.dtype("d"))
    fparam=np.array(fcoeff,dtype=np.dtype("d"))



    if isinstance(R, float) or isinstance(R, int):
        R=float(R)
        if rcut is None:
            rcut=2*R
        if zcut is None:
            zcut=2*R

        ret=[R,0]
        ret[1]=_vcirc_disc(R=R,zlaw=izdens,sigma0=sigma0, checkrd=checkrd, checkfl=checkfl,rparam=rparam,fparam=fparam, toll=toll,rcut=rcut,zcut=zcut)

        return np.array(ret)

    elif isinstance(R, list) or isinstance(R, tuple) or isinstance(R, np.ndarray):
        if rcut is None:
            rcut=2*np.max(R)
        if zcut is None:
            zcut=2*np.max(R)

        R=np.array(R,dtype=np.dtype("d"))
        nlenR=len(R)
        ret=np.array(_vcirc_disc_array(R=R, nlen=nlenR,zlaw=izdens,sigma0=sigma0, checkrd=checkrd, checkfl=checkfl, rparam=rparam,fparam=fparam, toll=toll,rcut=rcut,zcut=zcut))
        return  ret

    else:
        raise ValueError('R needs to be a float a int, an numpy array a tuple or a list.')



#######
#Vcirc Thin
cdef double _vcirc_disc_thin(double R, double sigma0, double checkrd,  double[:] rparam,  double toll, double rcut):
    """Vcirc of a razor thin disc disc for a single value of R
    :param R: Radial coordinate  (float or int)
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param rparam:  Params of the radial surface density law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :return: Vcirc at R in km/s
    """

    cdef:
        double G=4.518359396265313e-39 #kpc^3/(msol s^2)
        double kpc_to_km=3.08567758e16 #kpc_to_km
        double cost=(-4*G*sigma0*sqrt(R))
        #cost has 4 instead of 8 because the int^inf_-inf delta(=1, but  in the formulation
        #of integrand function we exploit the symmetry and we integrate  2*int^inf_0 dz, so
        #in this case we need to divide the final result by 2.
        double intvcirc, vc2, vc



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_vcirc as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zdirac')

    cdef:
        double d0=rparam[0]
        double d1=rparam[1]
        double d2=rparam[2]
        double d3=rparam[3]
        double d4=rparam[4]
        double d5=rparam[5]
        double d6=rparam[6]
        double d7=rparam[7]
        double d8=rparam[8]
        double d9=rparam[9]

    intvcirc=quad(fintegrand,0.,rcut, args=(R,checkrd,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9), epsabs=toll, epsrel=toll,points=(0,R))[0]

    vc2=cost*intvcirc

    if vc2>0:
        vc=sqrt(vc2) #vc in kpc/s
    else:
        vc=-sqrt(-vc2) #vc in kpc/s


    return vc*kpc_to_km #vc in km/s


cpdef double[:,:] _vcirc_disc_thin_array(double[:] R, int nlen, double sigma0, double checkrd, double[:] rparam,  double toll, double rcut):
    """Vcirc for a razor-thin disc for an array of R
    :param R: Radial coordinate  (list)
    :param nlen: length of the array R and Z (they should have the same dimension)
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param rparam:  Params of the radial surface density law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :return: Vcirc at R in km/s
    """

    cdef:
        double G=4.518359396265313e-39 #kpc^3/(msol s^2)
        double kpc_to_km=3.08567758e16 #kpc_to_km
        double cost=-(4*G*sigma0)
        #cost has 4 instead of 8 because the int^inf_-inf delta(=1, but  in the formulation
        #of integrand function we exploit the symmetry and we integrate  2*int^inf_0 dz, so
        #in this case we need to divide the final result by 2.
        double intvcirc, vc2, vc, R_tmp
        double[:,:] ret=np.empty((nlen,2), dtype=np.dtype("d"))
        int i



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_vcirc as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_vcirc_zdirac')


    cdef:
        double d0=rparam[0]
        double d1=rparam[1]
        double d2=rparam[2]
        double d3=rparam[3]
        double d4=rparam[4]
        double d5=rparam[5]
        double d6=rparam[6]
        double d7=rparam[7]
        double d8=rparam[8]
        double d9=rparam[9]


    for  i in range(nlen):

        R_tmp=R[i]

        ret[i,0]=R_tmp
        intvcirc=quad(fintegrand,0.,rcut, args=(R_tmp,checkrd,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9), epsabs=toll, epsrel=toll,points=(0,R_tmp))[0]

        vc2=cost*intvcirc*sqrt(R_tmp)

        if vc2>0:
            vc  =   sqrt(vc2) #vc in kpc/s
        else:
            vc  =   -sqrt(-vc2) #vc in kpc/s


        ret[i,1]=vc*kpc_to_km #vc in km/s

    return ret


cpdef vcirc_disc_thin(R, sigma0, rcoeff, rlaw='epoly', rcut=None, toll=1e-4):
    """Wrapper for the vcirc of a razor-thin disc

    :param R: Radial coordinates
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param rcoeff: Params of the radial surface density law
    :param rlaw: Radial surface density law
    :param rcut: Radial cut of the density
    :param toll: Relative tollerance for quad and nquad
    :return:  Vcirc at R in km/s
    """


    #Rdens
    if rlaw in checkrd_dict: checkrd=checkrd_dict[rlaw]
    else: raise NotImplementedError('Dens law %s not implmented'%rlaw)


    rparam=np.array(rcoeff,dtype=np.dtype("d"))



    if isinstance(R, float) or isinstance(R, int):
        R=float(R)
        if rcut is None:
            rcut=2*R


        ret=[R,0]
        ret[1]=_vcirc_disc_thin(R=R, sigma0=sigma0, checkrd=checkrd, rparam=rparam, toll=toll,rcut=rcut)

        return np.array(ret)

    elif isinstance(R, list) or isinstance(R, tuple) or isinstance(R, np.ndarray):
        if rcut is None:
            rcut=2*np.max(R)

        R=np.array(R,dtype=np.dtype("d"))
        nlenR=len(R)
        ret=np.array(_vcirc_disc_thin_array(R=R, nlen=nlenR, sigma0=sigma0, checkrd=checkrd,  rparam=rparam, toll=toll,rcut=rcut))
        return  ret

    else:
        raise ValueError('R needs to be a float a int, an numpy array a tuple or a list.')
