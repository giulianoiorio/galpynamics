#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double dens_core_nfw(double m, void * params) nogil
cdef double psi_core_nfw(double d0, double rs, double rc, double n, double m,  double toll) nogil
cdef double integrand_core_nfw(int nn, double *data) nogil
cdef double  _potential_core_nfw(double R, double Z, double mcut, double d0, double rs, double rc, double n, double e, double toll)
cdef double[:,:]  _potential_core_nfw_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double rc, double n, double e, double toll)
cdef double[:,:]  _potential_core_nfw_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double rc, double n,  double e, double toll)
cdef double vcirc_integrand_core_nfw(int n, double *data) nogil
