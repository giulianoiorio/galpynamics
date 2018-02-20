#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow, asinh, tanh, cosh, fabs
import numpy as np

cdef double PI=3.14159265358979323846

hwhm_fact = {'exp': 0.693, 'sech2': 0.881, 'gau': 1.177, 'dirac': 0} #Factor to pass from the Zd to the HWHM

#all normlised to the integral from -infty to infty.

cdef double zexp(double z, double zd) nogil:
    """
    Z=exp(-z/zd)/(2*zd)
    :param z:
    :param zd: scale height
    :return:
    """

    cdef:
        double norm, densz

    norm=(1/(2*zd))
    densz=exp(-fabs(z/zd))

    return norm*densz

cdef double zexp_der(double z, double zd) nogil:
    """
    Z=-exp(-z/zd)/(2*zd^2)=-zexp/zd
    :param z:
    :param zd: scale height
    :return:
    """
    cdef double zder=-zexp(z,zd)/(zd)

    return zder


cdef double zgau(double z, double zd) nogil:
    """
    Gau(z)=Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd)
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double norm, densz

    #3D dens
    norm=(1/(sqrt(2*PI)*zd))
    densz=exp(-0.5*(z/zd)*(z/zd))

    return densz*norm

cdef double zgau_der(double z, double zd) nogil:
    """
    Gau_der(z)=-Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd) * (z/zd^2) = -Gau(z)*(z/zd^2)
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double func, der_fact

    func=zgau(z, zd)
    der_fact=-z/(zd*zd)


    return func*der_fact

cdef double zsech2(double z, double zd) nogil:
    """
    Sech2(z)=(Sech(z/zd))^2 / ()
    :param z:
    :param zd: scale height
    :return:
    """

    cdef:
        double norm, densz

    norm=(1/(2*zd))
    densz=(1/(cosh(z/zd)) ) *  (1/(cosh(z/zd)) )

    return norm*densz

cdef double zsech2_der(double z, double zd) nogil:
    """
    Sech2_der(z)=-(Sech(z/zd))^2 / (2*zd) *  2*Tanh(z/zd)/zd=  - Sech2(z/zd) * 2*Tanh(z/zd)/zd
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double func, der_fact

    func=zsech2(z, zd)
    der_fact=-2*tanh(z/zd)/zd


    return func*der_fact

cdef double zdirac(double z, double zd) nogil:
    """
    delta(z)=1 if z==0, 0 otherwise.
    :param z:
    :param zd: nothing
    :return:
    """
    cdef:
        double ret

    if z==0:
        ret=1
    else:
        ret=0

    return ret



def  pyzexp(z, zd):
    """
    Z=exp(-z/zd)/(2*zd)
    :param z:
    :param zd: scale height
    :return:
    """

    norm=(1/(2*zd))
    densz=np.exp(-np.abs(z/zd))

    return norm*densz

def pyzexp_der(z,  zd):
    """
    Z=-exp(-z/zd)/(2*zd^2)=-zexp/zd
    :param z:
    :param zd: scale height
    :return:
    """
    zder=-pyzexp(z,zd)/(zd)

    return zder


def  pyzgau( z, zd):
    """
    Gau(z)=Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd)
    :param z:
    :param zd: dispersion
    :return:
    """


    #3D dens
    norm=(1/(np.sqrt(2*PI)*zd))
    densz=np.exp(-0.5*(z/zd)*(z/zd))

    return densz*norm

def  pyzgau_der(z, zd):
    """
    Gau_der(z)=-Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd) * (z/zd^2) = -Gau(z)*(z/zd^2)
    :param z:
    :param zd: dispersion
    :return:
    """


    func=pyzgau(z, zd)
    der_fact=-z/(zd*zd)


    return func*der_fact

def pyzsech2(z, zd):
    """
    Sech2(z)=(Sech(z/zd))^2 / ()
    :param z:
    :param zd: scale height
    :return:
    """



    norm=(1/(2*zd))
    densz=(1/(np.cosh(z/zd)) ) *  (1/(np.cosh(z/zd)) )

    return norm*densz

def pyzsech2_der(z,  zd):
    """
    Sech2_der(z)=-(Sech(z/zd))^2 / (2*zd) *  2*Tanh(z/zd)/zd=  - Sech2(z/zd) * 2*Tanh(z/zd)/zd
    :param z:
    :param zd: dispersion
    :return:
    """



    func=pyzsech2(z, zd)
    der_fact=-2*tanh(z/zd)/zd


    return func*der_fact

def pyzdirac(z, zd):
    """
    delta(z)=1 if z==0, 0 otherwise.
    :param z:
    :param zd: nothing
    :return:
    """

    if isinstance(z,float) or isinstance(z, int):
        z=np.array([z,])

    return np.where(z==0,1,0)

zfunc_dict = {'exp': pyzexp, 'sech2': pyzsech2, 'gau': pyzgau, 'dirac': pyzdirac}
