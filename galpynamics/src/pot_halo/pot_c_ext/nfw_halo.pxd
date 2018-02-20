#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double psi_nfw(double d0,double rs,double m) nogil
cdef double integrand_hnfw(int n, double *data) nogil
cdef double _potential_nfw(double R, double Z, double mcut, double d0, double rs, double e, double toll)
cdef double[:,:]  _potential_nfw_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double e, double toll)
cdef double[:,:]  _potential_nfw_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double e, double toll)
cdef double vcirc_integrand_nfw(int n, double *data) nogil