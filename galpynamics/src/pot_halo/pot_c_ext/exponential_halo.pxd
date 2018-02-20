cdef double psi_exponential(double d0, double rb, double m) nogil
cdef double integrand_exponential(int n, double *data) nogil
cdef double  _potential_exponential(double R, double Z, double mcut, double d0, double rb, double e, double toll)
cdef double[:,:]  _potential_exponential_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rb, double e, double toll)
cdef double[:,:]  _potential_exponential_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rb, double e, double toll)
cdef double vcirc_integrand_exponential(int n, double *data) nogil