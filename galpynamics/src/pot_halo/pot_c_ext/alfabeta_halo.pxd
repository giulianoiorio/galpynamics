#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double psi_alfabeta(double d0, double alpha, double beta, double rc, double m) nogil
cdef double integrand_alfabeta(int n, double *data) nogil
cdef double  _potential_alfabeta(double R, double Z, double mcut, double d0, double alfa, double beta, double rc, double e, double toll)
cdef double[:,:]  _potential_alfabeta_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double alfa, double beta, double rc, double e, double toll)
cdef double[:,:]  _potential_alfabeta_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double alfa, double beta,  double rc, double e, double toll)
cdef double vcirc_integrand_alfabeta(int n, double *data) nogil