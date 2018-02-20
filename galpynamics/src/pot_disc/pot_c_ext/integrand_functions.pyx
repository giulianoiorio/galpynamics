#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, fabs, cosh
from cython_gsl cimport *
from .rdens_law cimport poly_exponential, gaussian, fratlaw
from .rflare_law cimport poly_flare, constant, tanh_flare, asinh_flare
from scipy._lib._ccallback import LowLevelCallable
from scipy.integrate import nquad, quad
cimport numpy as np
import numpy as np
import ctypes
cdef double PI=3.14159265358979323846
from .model_option import checkrd_dict, checkfl_dict

#checkrd: 1-poly_exponential
#checkfl: 0-constat, 1-poly



######
#ZEXP
cdef double zexp(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Exp(-l/zd)/(2zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1: densr= poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==2: densr= fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==3: densr= gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    #Flare law
    if checkfli==0 : zd=constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==1: zd=poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==2: zd=asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==3: zd=tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)


    #3D dens
    norm=(1/(2*zd))
    densz=exp(-fabs(l/zd))

    #return 3.
    return densr*densz*norm


#Potential calc
cdef double integrand_zexp(int n, double *data) nogil:
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

    n=26

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[4]
        double checkfl = data[5]
        double d0 = data[6] #Rd
        double d1 = data[7]
        double d2 = data[8]
        double d3 = data[9]
        double d4 = data[10]
        double d5 = data[11]
        double d6 = data[12]
        double d7 = data[13]
        double d8 = data[14]
        double d9 = data[15]
        double f0 = data[16]
        double f1 = data[17]
        double f2 = data[18]
        double f3 = data[19]
        double f4 = data[20]
        double f5 = data[21]
        double f6 = data[22]
        double f7 = data[23]
        double f8 = data[24]
        double f9 = data[25]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zexp(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##########

######
#ZGAU
cdef double zgau(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1: densr= poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==2: densr= fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==3: densr= gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    #Flare law
    if checkfli==0 : zd=constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==1: zd=poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==2: zd=asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==3: zd=tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)


    #3D dens
    norm=(1/(sqrt(2*PI)*zd))
    densz=exp(-0.5*(l/zd)*(l/zd))

    #return 3.
    return densr*densz*norm


#Potential calc
cdef double integrand_zgau(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)

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

    n=26

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[4]
        double checkfl = data[5]
        double d0 = data[6] #Rd
        double d1 = data[7]
        double d2 = data[8]
        double d3 = data[9]
        double d4 = data[10]
        double d5 = data[11]
        double d6 = data[12]
        double d7 = data[13]
        double d8 = data[14]
        double d9 = data[15]
        double f0 = data[16]
        double f1 = data[17]
        double f2 = data[18]
        double f3 = data[19]
        double f4 = data[20]
        double f5 = data[21]
        double f6 = data[22]
        double f7 = data[23]
        double f8 = data[24]
        double f9 = data[25]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zgau(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##########

######
#Z
cdef double zsech2(double u, double l, double checkrd, double checkfl, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9) nogil:
    """Vertical law: Sech(-l/zd)^2
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param checkrd: Option to choice the radial surface density law
    :param checkfl: Option to choice the flaring density law
    :param d0-d9: Params of the radial surface density law
    :param f0-f9: Params of the radial flaring law
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz
        int checkrdi=<int> checkrd, checkfli=<int> checkfl

    #Dens law
    if checkrdi==1: densr= poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==2: densr= fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==3: densr= gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)


    #Flare law
    if checkfli==0 : zd=constant(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==1: zd=poly_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==2: zd=asinh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)
    elif checkfli==3: zd=tanh_flare(u, f0, f1, f2,f3,f4,f5, f6, f7, f8, f9)



    #3D dens
    norm=(1/(2*zd))
    densz=(1/(cosh(l/zd)) ) *  (1/(cosh(l/zd)) )

    #return 3.
    return densr*densz*norm


#Potential calc
cdef double integrand_zsech2(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Sech(-l/zd)^2

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

    n=26

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[4]
        double checkfl = data[5]
        double d0 = data[6] #Rd
        double d1 = data[7]
        double d2 = data[8]
        double d3 = data[9]
        double d4 = data[10]
        double d5 = data[11]
        double d6 = data[12]
        double d7 = data[13]
        double d8 = data[14]
        double d9 = data[15]
        double f0 = data[16]
        double f1 = data[17]
        double f2 = data[18]
        double f3 = data[19]
        double f4 = data[20]
        double f5 = data[21]
        double f6 = data[22]
        double f7 = data[23]
        double f8 = data[24]
        double f9 = data[25]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zsech2(u,l, checkrd, checkfl, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##############


#Delta dirac
######
cdef double zdirac(double u,  double checkrd, double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9) nogil:
    """Vertical law: delta(Z=0)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param checkrd: Option to choice the radial surface density law
    :param d0-d9: Params of the radial surface density law
    :return: The 2D density at R=u
    """

    cdef:
        double zd, norm, densr, densz
        int checkrdi=<int> checkrd

    #Dens law
    if checkrdi==1: densr= poly_exponential(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==2: densr= fratlaw(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
    elif checkrdi==3: densr= gaussian(u, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)

    #return 3.
    return densr

cdef double integrand_zdirac(int n, double *data) nogil:
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

    n=14

    cdef:
        double u = data[0] #R intengration variable
        double R = data[1]
        double Z = data[2]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double checkrd = data[3]
        double d0 = data[4]
        double d1 = data[5]
        double d2 = data[6]
        double d3 = data[7]
        double d4 = data[8]
        double d5 = data[9]
        double d6 = data[10]
        double d7 = data[11]
        double d8 = data[12]
        double d9 = data[13]
        double x, y


    x=(R*R + u*u + Z*Z)/(2*R*u)
    y=2/(x+1)
    dens=zdirac(u,checkrd, d0,d1,d2,d3,d4,d5,d6,d7,d8,d9)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens

#######
#Potential THICK
cdef double _potential_disc(double R, double Z, int zlaw, double sigma0, double checkrd, double checkfl, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """Potential disc for a single value of R and Z
    :param R: Radial coordinate  (float or int)
    :param Z: Vertical coordinate (float or int)
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
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)/(sqrt(R))
        double intpot



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau')


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

    #print('d',d0,d1,d2,d3,d4,d5,d6,d7,d8,d9)
    #print('f',f0,f1,f2,f3,f4,f5,f6,f7,f8,f9)
    intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R,Z,checkrd,checkfl,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[Z],'epsabs':toll,'epsrel':toll})])[0]

    return cost*intpot

#array
cdef double[:,:] _potential_disc_array(double[:] R, double[:] Z, int nlen , int zlaw, double sigma0,  double checkrd, double checkfl, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """Potential disc for an array of R and Z
    :param R: Radial coordinate  (list)
    :param Z: Vertical coordinate (list)
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
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double intpot
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        int i



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau')

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


        ret[i,0]=R[i]
        ret[i,1]=Z[i]


        intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R[i],Z[i],checkrd,checkfl,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9),opts=[({'points':(0,R[i]),'epsabs':toll,'epsrel':toll}),({'points':(Z[i],),'epsabs':toll,'epsrel':toll})])[0]


        ret[i,2]=(cost/(sqrt(R[i])))*intpot

    return ret


#grid
cdef double[:,:] _potential_disc_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, int zlaw, double sigma0, double checkrd, double checkfl, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """Potential disc for a a grid in  R and Z
    :param R: Radial coordinate  (list)
    :param Z: Vertical coordinate (list)
    :param nlenR: length of the array R
    :param nlenZ: length of the array Z
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
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau')

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

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]


            intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R[i],Z[j],checkrd,checkfl,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9),opts=[({'points':(0,R[i]),'epsabs':toll,'epsrel':toll}),({'points':(Z[j],),'epsabs':toll,'epsrel':toll})])[0]

            ret[c,2]=(cost/(sqrt(R[i])))*intpot

            c+=1

    return ret
####


cpdef potential_disc(R, Z, sigma0, rcoeff, fcoeff, zlaw='gau', rlaw='epoly', flaw='poly', rcut=None, zcut=None, toll=1e-4, grid=False):
    """Wrapper for the potential of a disc

    :param R: Radial coordinates
    :param Z: Vertical coordinates
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param rcoeff: Params of the radial surface density law
    :param fcoeff: Params of the radial flaring law
    :param zlaw: Vertical density law (exp, gau or sech2)
    :param rlaw: Radial surface density law
    :param flaw: Radial flaring law
    :param rcut: Radial cut of the density
    :param zcut: Vertical cut of the density
    :param toll: Relative tollerance for quad and nquad
    :param grid: If True, use the coordinates R and Z to make a grid of values
    :return:  Potential at R and Z in kpc/Myr
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
        if isinstance(Z, float) or isinstance(Z, int):
            R=float(R)
            Z=float(Z)
            if rcut is None:
                rcut=2*R
            if zcut is None:
                zcut=2*Z

            ret=[R,Z,0]
            ret[2]=_potential_disc(R=R,Z=Z,zlaw=izdens,sigma0=sigma0, checkrd=checkrd, checkfl=checkfl,rparam=rparam,fparam=fparam, toll=toll,rcut=rcut,zcut=zcut)

            return np.array(ret)
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if rcut is None:
            rcut=2*np.max(R)
        if zcut is None:
            zcut=2*np.max(np.abs(Z))

        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            nlenR=len(R)
            nlenZ=len(Z)
            return np.array(_potential_disc_grid(R=R,Z=Z,nlenR=nlenR, nlenZ=nlenZ,zlaw=izdens,sigma0=sigma0, checkrd=checkrd, checkfl=checkfl, rparam=rparam,fparam=fparam, toll=toll,rcut=rcut,zcut=zcut))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_disc_array(R=R,Z=Z,nlen=nlen,zlaw=izdens,sigma0=sigma0, checkrd=checkrd, checkfl=checkfl, rparam=rparam,fparam=fparam, toll=toll,rcut=rcut,zcut=zcut))
        else:
            raise ValueError('R and Z have different dimension')
###########

##########
#Potential thin
cdef double _potential_disc_thin(double R, double Z, double sigma0, double checkrd, double[:] rparam, double toll, double rcut):
    """Potential for a razor-thin  disc for a single value of R and Z
    :param R: Radial coordinate  (float or int)
    :param Z: Vertical coordinate (float or int)
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param rparam:  Params of the radial surface density law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :return: Potential at R and Z in kpc/Myr
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)/(sqrt(R))
        double intpot



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_zdirac')

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


    intpot=quad(fintegrand, 0., rcut, args=(R,Z,checkrd,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9), epsabs=toll, epsrel=toll)[0]


    return cost*intpot

#array
cdef double[:,:] _potential_disc_thin_array(double[:] R, double[:] Z, int nlen, double sigma0,  double checkrd, double[:] rparam, double toll, double rcut):
    """Potential for a razor-thin disc for an array of R and Z
    :param R: Radial coordinate  (list)
    :param Z: Vertical coordinate (list)
    :param nlen: length of the array R and Z (they should have the same dimension)
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param rparam:  Params of the radial surface density law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :return: Potential at R and Z in kpc/Myr
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double intpot
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        int i



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_zdirac')

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


        ret[i,0]=R[i]
        ret[i,1]=Z[i]


        intpot=quad(fintegrand, 0., rcut, args=(R[i],Z[i],checkrd,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9), epsabs=toll, epsrel=toll)[0]


        ret[i,2]=(cost/(sqrt(R[i])))*intpot

    return ret

#grid
cdef double[:,:] _potential_disc_thin_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double sigma0, double checkrd, double[:] rparam, double toll, double rcut):
    """Potential for a razor-thin disc for a a grid in  R and Z
    :param R: Radial coordinate  (list)
    :param Z: Vertical coordinate (list)
    :param nlenR: length of the array R
    :param nlenZ: length of the array Z
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param checkrd: number to choice the radial surface density law
    :param rparam:  Params of the radial surface density law
    :param toll: Relative tollerance for quad and nquad
    :param rcut: Radial cut of the density
    :return: Potential at R and Z in kpc/Myr
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_zdirac')

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

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]


            intpot=quad(fintegrand, 0., rcut, args=(R[i],Z[j],checkrd,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9), epsabs=toll, epsrel=toll)[0]

            ret[c,2]=(cost/(sqrt(R[i])))*intpot

            c+=1

    return ret

cpdef potential_disc_thin(R, Z, sigma0, rcoeff, rlaw='epoly', rcut=None, toll=1e-4, grid=False):
    """Wrapper for the potential of razor-thin disc

    :param R: Radial coordinates
    :param Z: Vertical coordinates
    :param sigma0: Value of the central disc surface density in Msun/kpc2
    :param rcoeff: Params of the radial surface density law
    :param rlaw: Radial surface density law
    :param rcut: Radial cut of the density
    :param toll: Relative tollerance for quad and nquad
    :param grid: If True, use the coordinates R and Z to make a grid of values
    :return:  Potential at R and Z  in kpc/Myr
    """
    #Rdens
    if rlaw in checkrd_dict: checkrd=checkrd_dict[rlaw]
    else: raise NotImplementedError('Dens law %s not implmented'%rlaw)

    rparam=np.array(rcoeff,dtype=np.dtype("d"))

    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            R=float(R)
            Z=float(Z)
            if rcut is None: rcut=2*R

            ret=[R,Z,0]
            ret[2]=_potential_disc_thin(R=R,Z=Z, sigma0=sigma0, checkrd=checkrd, rparam=rparam, toll=toll,rcut=rcut)

            return np.array(ret)
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if rcut is None: rcut=2*np.max(R)

        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            nlenR=len(R)
            nlenZ=len(Z)
            return np.array(_potential_disc_thin_grid(R=R,Z=Z,nlenR=nlenR, nlenZ=nlenZ, sigma0=sigma0, checkrd=checkrd,  rparam=rparam, toll=toll,rcut=rcut))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_disc_thin_array(R=R,Z=Z,nlen=nlen,sigma0=sigma0, checkrd=checkrd, rparam=rparam, toll=toll,rcut=rcut))
        else:
            raise ValueError('R and Z have different dimension')



