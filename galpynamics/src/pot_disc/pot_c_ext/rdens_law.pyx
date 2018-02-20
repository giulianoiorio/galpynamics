#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow
import numpy as np

ctypedef double (*f_type)(double, double[:], int) nogil

#All normalizes over sigma0

cdef double poly_exponential(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    """

    :param R: variable
    :param a0: Radial scale length in Kpc
    :param a1: coeff-0 polynomial
    :param a2: coeff-1 polynomial
    :param a3: coeff-2 polynomial
    :param a4: coeff-3 polynomial
    :param a5: coeff-4 polynomial
    :param a6: coeff-5 polynomial
    :param a7: coeff-6 polynomial
    :param a8: coeff-7 polynomial
    :param a9: coeff-8 polynomial
    :return:
    """


    cdef:
        double res, recursiveR, Rd

    Rd=a0


    res=a1

    recursiveR=R
    res+=a2*recursiveR
    recursiveR=recursiveR*R
    res+=a3*recursiveR
    recursiveR=recursiveR*R
    res+=a4*recursiveR
    recursiveR=recursiveR*R
    res+=a5*recursiveR
    recursiveR=recursiveR*R
    res+=a6*recursiveR
    recursiveR=recursiveR*R
    res+=a7*recursiveR
    recursiveR=recursiveR*R
    res+=a8*recursiveR
    recursiveR=recursiveR*R
    res+=a9*recursiveR
    res=res #Normalised over sigma0

    return res*exp(-R/Rd)

cdef double poly_exponential_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    #NB a0=Rd
    #if exp=Exp(-R/Rd) and poly=Poly(a1...a9), the derivative of poly*exp is polyder*exp + poly*expder
    #The derivative is
    cdef:
        double  polyder_exp, poly_expder
        double exp, exp_der
        double Rd=a0

    polyder_exp= poly_exponential(R, Rd, a2, 2.*a3, 3.*a4, 4.*a5, 5.*a6, 6.*a7, 7.*a8, 8.*a9, 0.)
    poly_expder= -poly_exponential(R, Rd, a1, a2, a3, a4, a5, a6, a7 , a8 , a9)/Rd

    return  polyder_exp + poly_expder


cdef double gaussian(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:


    cdef:
        double sigmad=a0
        double R0=a1
        double earg, sigmad2, eargnorm,rexp, rexpnorm

    sigmad2=sigmad*sigmad

    earg=(R-R0)*(R-R0)/(sigmad2)
    eargnorm=(R0*R0)/(sigmad2)

    rexp=exp(-0.5*earg)
    rexpnorm=exp(-0.5*eargnorm)

    return rexp/rexpnorm

cdef double gaussian_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    #Def Gaunorm=Gau/Gau(0)= Gau/Gau0*(R-R0)/s^2=Gaunorm * (R-R0)/s^2

    cdef:
        double sigmad=a0
        double R0=a1
        double gaunorm, derfact

    gaunorm=gaussian(R, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    derfact=-(R-R0)/(sigmad*sigmad)

    return gaunorm*derfact


cdef double fratlaw(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    cdef:
        double Rd=a0
        double Rd2=a1
        double alpha=a2
        double mult1, mult1a, mult2, xd, xd2


    xd=R/Rd
    xd2=R/Rd2
    mult1=(1+xd2)
    mult1a=pow(mult1,alpha)
    mult2=exp(-xd)

    return mult1a*mult2

cdef double fratlaw_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    #der fratlaw= alpha/(Rd2*(1+R/Rd2)) *fratlaw  +  - fratlaw/Rd

    cdef:
        double Rd=a0
        double Rd2=a1
        double alpha=a2
        double func_original
        double numA, denA, parta, partb

    func_original=fratlaw(R, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    xd2=R/Rd2
    mult=(1+xd2)

    numA=alpha*func_original
    denA=Rd2*mult
    parta=numA/denA
    partb=-func_original/Rd


    return parta+partb


cpdef rdens(R, checkrd,  double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9):
    """
    Return Sup density
    :param R: Cilindrical radius
    :param rparam:

    """

    cdef:
        int i
        double Rtmp

    checkrdi=int(checkrd)
    #Flare law
    if checkrdi==1:
        rdens_func        = poly_exponential
    elif checkrdi==2:
        rdens_func        = fratlaw
    elif checkrdi==3:
        rdens_func        = gaussian

    if isinstance(R, int) or isinstance(R, float):
        ret=np.array([[R, 0]])

        ret[0,1]=rdens_func(R, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)

    elif isinstance(R, list) or isinstance(R, tuple) or isinstance(R, np.ndarray):

        ret=np.zeros(shape=(len(R),2))
        ret[:,0]=R

        for i in range(len(R)):
            Rtmp=ret[i,0]
            ret[i,1]= rdens_func(Rtmp, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)


    return ret




