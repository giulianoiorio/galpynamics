cdef double psi_valy(double d0, double rb, double m) nogil
cdef double integrand_valy(int n, double *data) nogil
cdef double  _potential_valy(double R, double Z, double mcut, double d0, double rb, double e, double toll)
cdef double[:,:]  _potential_valy_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rb, double e, double toll)
cdef double[:,:]  _potential_valy_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rb, double e, double toll)
cdef double vcirc_integrand_valy(int n, double *data) nogil