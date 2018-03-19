#cython: language_level=3, boundscheck=False
from __future__ import print_function
import pip
import time

#Check Cython installation
print('Checking Ctyhon')
try:
    import Cython
    cyv=Cython.__version__
    print('OK! (Version %s)'%cyv)
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','Cython'])

#Check CythonGSL installation
print('Checking CtyhonGSL')
try:
    import cython_gsl
    print('OK!')
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','CythonGSL'])

#Check Scipy>1.0 installation
print('Checking Scipy>1.0')
try:
    import scipy
    scv=scipy.__version__
    scvl=scv.split('.')
    if int(scvl[0])>0 or int(scvl[1])>19:
        print('OK! (Version %s)'%scv)
    else:
        print('Version %s too old. I will install the lastest version' % scv)
        pip.main(['install','scipy'])
except:
    print('Scipy is not present, I will install it for you my lord')
    pip.main(['install','scipy'])





from setuptools import setup
import shutil
import os
from Cython.Distutils import build_ext
from distutils.core import Extension
from Cython.Build import cythonize
import sysconfig
import numpy
import cython_gsl
import sys

if sys.version_info[0]==2:
    #time.sleep(5)
    cmdclass_option = {}
    print('You are using Python2, what a shame!')
    #raise ValueError('You are using Python2, what a shame! Download Python3 to use this module. \n If you are using anaconda you can install a python3 virtual env just typing:\n "conda create -n yourenvname python=3.6 anaconda". \n Then you can activate the env with the bash command  "source activate yourenvname"')

elif sys.version_info[0]==3:
    print('You are using Python3, you are a wise person!')

    def get_ext_filename_without_platform_suffix(filename):
        name, ext = os.path.splitext(filename)
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        if ext_suffix == ext:
            return filename

        ext_suffix = ext_suffix.replace(ext, '')
        idx = name.find(ext_suffix)

        if idx == -1:
            return filename
        else:
            return name[:idx] + ext


    class BuildExtWithoutPlatformSuffix(build_ext):
        def get_ext_filename(self, ext_name):
            filename = super().get_ext_filename(ext_name)
            return get_ext_filename_without_platform_suffix(filename)
    cmdclass_option = {'build_ext': BuildExtWithoutPlatformSuffix}
else:
    raise ValueError('You are not using neither Python2 nor Python3, probably you are a time traveller from the Future or from the Past')

#cython gsl
cy_gsl_lib=cython_gsl.get_libraries()
cy_gsl_inc=cython_gsl.get_include()
cy_gsl_lib_dic=cython_gsl.get_library_dir()
#cython
cy_gsl_inc_cy=cython_gsl.get_cython_include_dir()
#numpy
np_inc=numpy.get_include()


gh=['galpynamics/src/pot_halo/pot_c_ext/general_halo.pyx']
gh_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/general_halo',sources=gh)

ih=['galpynamics/src/pot_halo/pot_c_ext/isothermal_halo.pyx']
ih_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/isothermal_halo',sources=ih)

infw=['galpynamics/src/pot_halo/pot_c_ext/nfw_halo.pyx']
infw_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/nfw_halo',sources=infw)

iab=['galpynamics/src/pot_halo/pot_c_ext/alfabeta_halo.pyx']
iab_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/alfabeta_halo',sources=iab,libraries=cy_gsl_lib,library_dirs=[cy_gsl_lib_dic],include_dirs=[cy_gsl_inc_cy])

ph=['galpynamics/src/pot_halo/pot_c_ext/plummer_halo.pyx']
ph_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/plummer_halo',sources=ph)

eh=['galpynamics/src/pot_halo/pot_c_ext/einasto_halo.pyx']
eh_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/einasto_halo',sources=eh,libraries=cy_gsl_lib,library_dirs=[cy_gsl_lib_dic],include_dirs=[cy_gsl_inc_cy])

exh=['galpynamics/src/pot_halo/pot_c_ext/exponential_halo.pyx']
exh_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/exponential_halo',sources=exh)


vh=['galpynamics/src/pot_halo/pot_c_ext/valy_halo.pyx']
vh_ext=Extension('galpynamics/src/pot_halo/pot_c_ext/valy_halo',sources=vh)

gd=['galpynamics/src/pot_disc/pot_c_ext/integrand_functions.pyx']
gd_ext=Extension('galpynamics/src/pot_disc/pot_c_ext/integrand_functions',libraries=cy_gsl_lib,library_dirs=[cy_gsl_lib_dic],include_dirs=[cy_gsl_inc_cy, np_inc],sources=gd)

rd=['galpynamics/src/pot_disc/pot_c_ext/rdens_law.pyx']
rd_ext=Extension('galpynamics/src/pot_disc/pot_c_ext/rdens_law',sources=rd)

fd=['galpynamics/src/pot_disc/pot_c_ext/rflare_law.pyx']
fd_ext=Extension('galpynamics/src/pot_disc/pot_c_ext/rflare_law',sources=fd)

zd=['galpynamics/src/pot_disc/pot_c_ext/zdens_law.pyx']
zd_ext=Extension('galpynamics/src/pot_disc/pot_c_ext/zdens_law',sources=zd)

vcirc=['galpynamics/src/pot_disc/pot_c_ext/integrand_vcirc.pyx']
vcirc_ext=Extension('galpynamics/src/pot_disc/pot_c_ext/integrand_vcirc', sources=vcirc,libraries=cy_gsl_lib,library_dirs=[cy_gsl_lib_dic],include_dirs=[cy_gsl_inc_cy, np_inc])

#ext_modules=cythonize([cy_ext,gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext])

#extra_compile_args = ['-std=c99']
#sturct_c_src=['galpynamics/src/pot_disc/pot_c_ext/struct.c']
#struct_c_ext = Extension('galpynamics/src/pot_disc/pot_c_ext/struct',
                     #sources=sturct_c_src,
                     #extra_compile_args=extra_compile_args
                     #)


ext_modules=cythonize([gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext,iab_ext,ph_ext,eh_ext,vh_ext,exh_ext,zd_ext,vcirc_ext])

setup(
		name='galpynamics',
		version='0.1.dev0',
		author='Giuliano Iorio',
		author_email='',
		url='',
        cmdclass=cmdclass_option,
		packages=['galpynamics','galpynamics/src','galpynamics/src/pot_halo','galpynamics/src/pot_halo/pot_c_ext','galpynamics/src/pardo','galpynamics/src/pot_disc', 'galpynamics/src/pot_disc/pot_c_ext', 'galpynamics/src/galpotential', 'galpynamics/src/discHeight', 'galpynamics/src/discHeight/c_ext' , 'galpynamics/src/fitlib' ],
        ext_modules=ext_modules,
        include_dirs=[np_inc,cython_gsl.get_include()],
        install_requires=['numpy>=1.9', 'scipy>=0.19', 'matplotlib','emcee']
)

'''
try:
    shutil.rmtree('build')
    shutil.rmtree('dist')
    shutil.rmtree('galpynamics.egg-info')
except:
    pass
'''
