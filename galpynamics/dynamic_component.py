#from .src.pot_halo.pot_halo import isothermal_halo, NFW_halo, alfabeta_halo, hernquist_halo, deVacouler_like_halo, plummer_halo
#from .src.pot_halo.pot_halo import einasto_halo, valy_halo, exponential_halo
#from .src.pot_disc import Exponential_disc, Frat_disc, Gaussian_disc, PolyExponential_disc

#DISCs
from .src.pot_disc.Exponential_disc import Exponential_disc
from .src.pot_disc.Gaussian_disc import Gaussian_disc
from .src.pot_disc.PolyExponential_disc import PolyExponential_disc
from .src.pot_disc.Frat_disc import Frat_disc
from .src.pot_disc.McMillan_disc import McMillan_disc

#HALOes
from .src.pot_halo.isothermal_halo import isothermal_halo
from .src.pot_halo.NFW_halo import NFW_halo
from .src.pot_halo.alfabeta_halo import alfabeta_halo
from .src.pot_halo.truncated_alfabeta_halo import truncated_alfabeta_halo
from .src.pot_halo.hernquist_halo import hernquist_halo
from .src.pot_halo.deVacouler_like_halo import deVacouler_like_halo
from .src.pot_halo.plummer_halo import plummer_halo
from .src.pot_halo.valy_halo import valy_halo
from .src.pot_halo.exponential_halo import exponential_halo
