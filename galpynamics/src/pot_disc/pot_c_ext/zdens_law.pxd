#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

#all normlised to the integral from -infty to infty.

cdef double zexp(double z, double zd) nogil
cdef double zexp_der(double z, double zd) nogil

cdef double zgau(double z, double zd) nogil
cdef double zgau_der(double z, double zd) nogil

cdef double zsech2(double z, double zd) nogil
cdef double zsech2_der(double z, double zd) nogil