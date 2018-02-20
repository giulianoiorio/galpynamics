#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double psi_iso(double d0, double rc, double m) nogil
cdef double integrand_hiso(int n, double *data) nogil
cdef double  _potential_iso(double R, double Z, double mcut, double d0, double rc, double e, double toll)
cdef double[:,:]  _potential_iso_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rc, double e, double toll)
cdef double[:,:]  _potential_iso_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rc, double e, double toll)
cdef double vcirc_integrand_iso(int n, double *data) nogil