#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double psi_truncated_alfabeta(double d0, double rs, double alfa, double beta, double rcut, double m,  double toll) nogil
cdef double integrand_truncated_alfabeta(int nn, double *data) nogil
cdef double  _potential_truncated_alfabeta(double R, double Z, double mcut, double d0, double rs, double alfa, double beta, double rcut, double e, double toll)
cdef double[:,:]  _potential_truncated_alfabeta_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double alfa, double beta, double rcut, double e, double toll)
cdef double[:,:]  _potential_truncated_alfabeta_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double alfa, double beta,  double rcut, double e, double toll)
cdef double vcirc_integrand_truncated_alfabeta(int n, double *data) nogil
