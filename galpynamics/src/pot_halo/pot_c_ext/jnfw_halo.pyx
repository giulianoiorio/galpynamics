#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
import numpy as np
cimport numpy as np

cdef double PI=3.14159265358979323846