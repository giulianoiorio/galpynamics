from __future__ import print_function
import galpynamics
from timeit import default_timer as timer
import numpy as np
import sys


R=np.linspace(0,15,100000)
Z=np.linspace(0,5,100000)

def test_isothermal_halo():
    
    d0=3
    rc=2
    mcut=100

    print('#'*40)
    print('TEST: Isothermal halo')
    sys.stdout.flush()

    e=0.
    try:
        a=galpynamics.isothermal_halo(d0,rc,e,mcut)
        t1=timer()
        a.potential(R,Z,grid=False, toll=1e-4,nproc=1)
        t2=timer()
        checks='Success'
        times=t2-t1
    except:
        checks='Fail'

    try:
        a=galpynamics.isothermal_halo(d0,rc,e,mcut)
        t1=timer()
        a.potential(R,Z,grid=False, toll=1e-4,nproc=2)
        t2=timer()
        timem=t2-t1
        checkm='Success'
    except:
        checkm='Fail'

    if checks=='Success' and checkm=='Success':
        print('Spherical halo: single-cpu: %s; multi-cpu(2): %s. Speed-UP:%.1fX'%(checks,checkm,times/timem))
    else:
        print('Spherical halo: single-cpu: %s; multi-cpu(2): %s.'%(checks,checkm))
    sys.stdout.flush()

    e = 0.5
    try:
        a = galpynamics.isothermal_halo(d0, rc, e, mcut)
        t1 = timer()
        a.potential(R, Z, grid=False, toll=1e-4, nproc=1)
        t2 = timer()
        times=t2-t1
        checks = 'Success'
    except:
        checks = 'Fail'

    try:
        a = galpynamics.isothermal_halo(d0, rc, e, mcut)
        t1 = timer()
        a.potential(R, Z, grid=False, toll=1e-4, nproc=2)
        t2 = timer()
        timem=t2-t1
        checkm = 'Success'
    except:
        checkm = 'Fail'


    if checks=='Success' and checkm=='Success':
        print('Oblate halo: single-cpu: %s; multi-cpu(2): %s. Speed-UP:%.1fX'%(checks,checkm,times/timem))
    else:
        print('Oblate halo: single-cpu: %s; multi-cpu(2): %s.'%(checks,checkm))
    sys.stdout.flush()

    print('#' * 40)
    sys.stdout.flush()



if __name__ == '__main__':
    test_isothermal_halo()
