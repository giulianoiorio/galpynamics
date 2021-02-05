from __future__ import print_function
import galpynamics
from timeit import default_timer as timer
import numpy as np
import sys
from termcolor import colored

#TODO Disc Test


def _check_parallel(checks, checkm, times, timem, idnans, idnanm):
    if checks:
        outmess=colored('Success','green')
        print('single-cpu: '+outmess+ ' (with %i nans)'%(np.sum(idnans)) +' in %.1e s'%times)
    else:
        outmess=colored('Fail','red')
        print('single-cpu: '+outmess)
    sys.stdout.flush()

    if checkm:
        outmess=colored('Success','green')
        print('Multi-cpu (2): ' +outmess+ ' (with %i nans)'%(np.sum(idnanm)) + ' in %.1e s'%timem)
    else:
        outmess=colored('Fail','red')
        print('Multi-cpu (2): '+outmess)
    sys.stdout.flush()

def _check_same_output(output1, output2):


    same_output=(output1==output2).all()
    if same_output:
        outmess=colored('Success','green')
        print('Same Output: '+outmess)
    else:
        outmess=colored('Fail','red')
        print('Same Output: '+outmess)
    sys.stdout.flush()


def test_halo_component(halo_component, neval=1000):

    print('*'*50)
    print('Test halo component: %s'%halo_component.name)
    print('*'*50)
    sys.stdout.flush()
    ngeval=int(np.sqrt(neval))
    nvel=100

    R=np.linspace(0.05,15, neval)
    Z=np.linspace(0.05,5, neval)
    Rg=np.linspace(0.05,15, ngeval)
    Zg=np.linspace(0.05,5, ngeval)
    Rvel=np.linspace(0.05,15, nvel)

    H=halo_component
    print('-'*50)
    print('Test potential estimate no grid (%i evalutations)'%neval)
    print('-'*50)
    sys.stdout.flush()

    times=0
    idnan_serial=0
    try:
        t1=timer()
        Pot_serial=H.potential(R,Z,grid=False, toll=1e-4,nproc=1)
        idnan_serial=np.isnan(Pot_serial)
        Pot_serial=np.where(idnan_serial,-999, Pot_serial)
        t2=timer()
        checks=True
        times=t2-t1
    except:
        checks=False

    timem=0
    idnan_parallel=0
    try:
        t1=timer()
        Pot_parallel=H.potential(R,Z,grid=False, toll=1e-4,nproc=2)
        idnan_parallel=np.isnan(Pot_parallel)
        Pot_parallel=np.where(idnan_parallel,-999, Pot_parallel)
        t2=timer()
        timem=t2-t1
        checkm=True
    except:
       checkm=False

    _check_parallel(checks, checkm, times, timem, idnan_serial, idnan_parallel)
    if checks and checkm: _check_same_output(Pot_serial, Pot_parallel)

    print('-'*50)
    print('Test potential estimate grid (%i evalutations)'%(ngeval*ngeval))
    print('-'*50)
    sys.stdout.flush()

    try:
        t1=timer()
        Pot_serial=H.potential(Rg, Zg, grid=True, toll=1e-4,nproc=1)
        idnan_serial=np.isnan(Pot_serial)
        Pot_serial=np.where(np.isnan(Pot_serial),-999, Pot_serial)
        t2=timer()
        checks=True
        times=t2-t1
    except:
        checks=False

    try:
        t1=timer()
        Pot_parallel=H.potential(Rg, Zg, grid=True, toll=1e-4, nproc=2)
        idnan_parallel=np.isnan(Pot_parallel)
        Pot_parallel=np.where(np.isnan(Pot_parallel),-999, Pot_parallel)
        t2=timer()
        timem=t2-t1
        checkm=True
    except:
        checkm=False

    _check_parallel(checks, checkm, times, timem, idnan_serial, idnan_parallel)
    if checks and checkm: _check_same_output(Pot_serial, Pot_parallel)

    print('-'*50)
    print('Test Vcirc estimate  (%i evalutations)'%nvel)
    print('-'*50)
    sys.stdout.flush()

    try:
        t1=timer()
        Vel_serial=H.vcirc(Rvel, toll=1e-4,nproc=1)
        idnan_serial=np.isnan(Vel_serial)
        Vel_serial=np.where(np.isnan(Vel_serial),-999, Vel_serial)
        t2=timer()
        checks=True
        times=t2-t1
    except:
        checks=False

    try:
        t1=timer()
        Vel_parallel=H.vcirc(Rvel, toll=1e-4, nproc=2)
        idnan_parallel=np.isnan(Vel_parallel)
        Vel_parallel=np.where(np.isnan(Vel_parallel),-999, Vel_parallel)
        t2=timer()
        timem=t2-t1
        checkm=True
    except:
        checkm=False

    _check_parallel(checks, checkm, times, timem, idnan_serial, idnan_parallel)
    if checks and checkm: _check_same_output(Vel_serial, Vel_parallel)

    print('*'*50)


#Isothermal halo
d0=3
rc=2
mcut=100
e=0.5
_to_test=galpynamics.isothermal_halo(d0=d0, rc=rc, mcut=mcut, e=e)
test_halo_component(_to_test, 50000)

#NFW halo
d0=3
rs=2
mcut=100
e=0.5
_to_test=galpynamics.NFW_halo(d0=d0, rs=rs, mcut=mcut, e=e)
test_halo_component(_to_test, 50000)

#core NFW halo
d0=3
n=0.5
rc=2
rs=5
mcut=100
e=0.5
_to_test=galpynamics.core_NFW_halo(d0=d0, rc=rc, n=n, rs=rs, mcut=mcut, e=e)
test_halo_component(_to_test, 1000)

#alfabeta halo
d0=3
rs=5
alfa=1.5
beta=3.2
mcut=100
e=0.5
_to_test=galpynamics.alfabeta_halo(d0=d0, rs=rs, alfa=alfa, beta=beta, mcut=mcut, e=e)
test_halo_component(_to_test, 50000)

#truncated alfabeta halo
d0=3
rs=5
alfa=1.5
beta=3.2
rcut=10
mcut=100
e=0.5
_to_test=galpynamics.truncated_alfabeta_halo(d0=d0, rs=rs, alfa=alfa, beta=beta, rcut=rcut, mcut=mcut, e=e)
test_halo_component(_to_test, 500)



#Plummer halo
rc=5
mass=1e10
mcut=100
e=0.5
_to_test=galpynamics.plummer_halo(rc=rc, mass=mass, mcut=mcut, e=e)
test_halo_component(_to_test, 50000)

#Einasto halo
d0=3
rs=5
n=6
e=0.5
mcut=100
_to_test=galpynamics.einasto_halo(d0=d0, rs=rs, n=n, e=e, mcut=mcut)
test_halo_component(_to_test, 50000)

#Valy Halo
d0=3
rb=3
e=0.5
mcut=100
_to_test=galpynamics.valy_halo(rb=rb, d0=d0, e=e, mcut=mcut)
test_halo_component(_to_test, 50000)

#Exponential Halo
d0=3
rb=3
e=0.5
mcut=100
_to_test=galpynamics.exponential_halo(d0=d0, rb=rb, e=e, mcut=mcut)
test_halo_component(_to_test, 50000)

#MWBinney11
_to_test=galpynamics.MWBinney11()
test_halo_component(_to_test, 100)

#MWMcMillan17
_to_test=galpynamics.MWMcMillan17()
test_halo_component(_to_test, 100)
