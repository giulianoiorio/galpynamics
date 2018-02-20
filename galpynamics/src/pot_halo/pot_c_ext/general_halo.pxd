#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False


cdef double xi(double m,double R,double Z, double e) nogil
cdef double m_calc(double R, double Z, double e) nogil
cdef double integrand_core(double m, double R, double Z, double e, double psi) nogil
cdef double potential_core(double e, double intpot, double psi) nogil
cdef double vcirc_core(double m, double R, double e) nogil

