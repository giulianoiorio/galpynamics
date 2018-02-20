#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double dn_func(double n) nogil
cdef double dens_einasto(double m, void * params) nogil
cdef double psi_einasto(double d0, double rs, double n, double m,  double toll) nogil
cdef double integrand_einasto(int nn, double *data) nogil
cdef double  _potential_einasto(double R, double Z, double mcut, double d0, double rs, double n, double e, double toll)
cdef double[:,:]  _potential_einasto_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double n, double e, double toll)
cdef double[:,:]  _potential_einasto_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double n,  double e, double toll)
cdef double vcirc_integrand_einasto(int n, double *data) nogil
