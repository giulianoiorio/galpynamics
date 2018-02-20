#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

ctypedef double (*f_type)(double, double[:], int) nogil
cdef double integrand_zexp(int n, double *data) nogil
cdef double integrand_zsech2(int n, double *data) nogil
cdef double integrand_zgau(int n, double *data) nogil
cdef double integrand_zdirac(int n, double *data) nogil




