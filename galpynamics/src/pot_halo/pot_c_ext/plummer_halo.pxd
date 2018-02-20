
cdef double psi_plummer(double d0, double rc, double m) nogil
cdef double integrand_plummer(int n, double *data) nogil
cdef double  _potential_plummer(double R, double Z, double mcut, double d0, double rc, double e, double toll)
cdef double[:,:]  _potential_plummer_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rc, double e, double toll)
cdef double[:,:]  _potential_plummer_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rc, double e, double toll)
cdef double vcirc_integrand_plummer(int n, double *data) nogil