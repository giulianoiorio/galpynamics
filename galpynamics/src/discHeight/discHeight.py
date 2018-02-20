from __future__ import division, print_function
from ..galpotential.galpotential import galpotential
from ..pot_disc.pot_disc import disc
from ..pot_halo.pot_halo import  halo
from .c_ext.utility import cdens
from .c_ext import fitlib as ft
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
label_size =18
mpl.rcParams.update({'figure.autolayout':True})
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['mathtext.default']='regular'

class discHeight(object):

    def __init__(self, disc_component, dynamic_components=None):

        self._check_components(components=disc_component)
        self.disc_component=disc_component
        if dynamic_components is not None:
            self._check_components(components=dynamic_components)
        self.dynamic_components=dynamic_components

    def _fixed_potential(self,R,Z,grid,nproc,Rcut,zcut,mcut,toll,external_potential=None):

        df=galpotential(dynamic_components=self.dynamic_components)
        self.external_potential=external_potential

        df.potential(R,Z,grid=grid,nproc=nproc, toll=toll, Rcut=Rcut, zcut=zcut, mcut=mcut,external_potential=external_potential)
        self.fixed_potential_grid=df.potential_grid

        return df

    def _calc_flaring(self,pot_grid, vdisp_func,zlaw,outdir='',plot=False, diagnostic=False, output=False, dzlimit=1e-10):

        #Calc_discH
        dens_array=cdens(pot_grid, disp=vdisp_func)
        tabzd, tabhw, c = ft.fitzprof(dens_array[:][:, [0, 1, 3]], dists=zlaw, outdir=outdir, plot=plot, diagnostic=diagnostic, output=output,dzlimit=dzlimit)

        return tabzd, tabhw

    def _fit_flaring(self,tabzd,zlaw,polydegree=4,diagnostic=False,outdir='',Rlimit=None):

        ftab,fitfunc=ft.fitflare(tab=tabzd,func=zlaw,polydegree=polydegree,outdir=outdir,outfile=diagnostic,plot=diagnostic,Rlimit=Rlimit)
        ftab=ftab[:-1]
        lenftab=len(ftab)



        if lenftab<3:
            ftabs=np.zeros(3)
            if lenftab==1:   ftabs[0]=ftab[0]
            elif lenftab==2: ftabs[:2] = ftab
            elif lenftab==3: ftabs[:3] = ftab
            ftab=ftabs
        else:
            pass


        return ftab,fitfunc

    def height(self, flaw='poly', zlaw='gau',  polyflare_degree=5, vdisp=10, Rpoints=30, Rinterval='linear', Rrange=(0.01,30), Zpoints=30, Zinterval='log', Zrange=(0,10), Niter=10, nproc=2, Rcut=None, zcut=None, mcut=None, flaretollabs=1e-4,flaretollrel=1e-4, inttoll=1e-4, external_potential=None, outdir='gasHeight', diagnostic=True, Rlimit=None):

        comp=self.disc_component
        vdisp_func=self._set_vdisp(vdisp)

        Zmin,Zmax=Zrange
        if (isinstance(Zpoints,int)):
            if Zinterval=='linear': Z=np.linspace(Zmin,Zmax,Zpoints)
            elif Zinterval=='log': Z=np.logspace(np.log10(Zmin+0.001),np.log10(Zmax+0.001),Zpoints)-0.001
        elif isinstance(Zpoints,list) or isinstance(Zpoints,tuple) or isinstance(Zpoints,np.ndarray):
            Z=Zpoints

        Rmin, Rmax= Rrange
        if (isinstance(Rpoints, int)):
            if Rinterval == 'linear':
                R = np.linspace(Rmin, Rmax, Rpoints)
            elif Rinterval == 'log':
                R = np.logspace(np.log10(Rmin + 0.001), np.log10(Rmax + 0.001), Rpoints) - 0.001
        elif isinstance(Zpoints, list) or isinstance(Zpoints, tuple) or isinstance(Zpoints, np.ndarray):
            R = Rpoints

        fig10 = plt.figure()
        ax10 = fig10.add_subplot(1, 1, 1)

        #Fixed potential
        print('//////////////////////////////////////////////////////////////////////////////')
        print('Calculating fixed potential')
        sys.stdout.flush()
        df_fix=self._fixed_potential(R=R, Z=Z, grid=True, nproc=nproc, Rcut=Rcut, zcut=zcut, mcut=mcut, toll=inttoll,external_potential=external_potential)
        fixed_potential=self.fixed_potential_grid
        print('Fixed potential Done')
        print('//////////////////////////////////////////////////////////////////////////////\n')
        sys.stdout.flush()

        if Rlimit=='max': Rlimit=R[-1]
        flare_max=[]

        print('//////////////////////////////////////////////////////////////////////////////')
        print('Iter-0: Massless disc')
        sys.stdout.flush()
        outfolder='/diagnostic/run0'
        tabzd,tabh=self._calc_flaring(pot_grid=fixed_potential,vdisp_func=vdisp_func,zlaw=zlaw, outdir=outdir + outfolder, plot=diagnostic, diagnostic=diagnostic, output=diagnostic)
        ftab,fitfunc=self._fit_flaring(tabzd=tabzd,zlaw=flaw,polydegree=polyflare_degree,diagnostic=diagnostic,outdir=outdir + outfolder+'/flare', Rlimit=Rlimit)
        oldtabzd=tabzd
        flare_max=np.max(tabzd[:,1])
        print('Iter-0: Done')
        print('//////////////////////////////////////////////////////////////////////////////\n')
        sys.stdout.flush()

        ax10.plot(tabzd[:,0],tabzd[:,1], '-o', color='gray')

        count=0
        max_residual_abs=1e10
        max_residual_rel=1e10
        while (count<=Niter) and (max_residual_abs>flaretollabs) and (max_residual_rel>flaretollrel):

            #new model
            self.disc_component=comp.change_flaring(flaw=flaw,zlaw=zlaw,polycoeff=ftab,h0=ftab[0], Rf=ftab[1], c=ftab[2], Rlimit=Rlimit)


            print('//////////////////////////////////////////////////////////////////////////////')
            sys.stdout.flush()
            print('Iter-%i:'%(count+1))
            sys.stdout.flush()
            outfolder = '/diagnostic/run%i'%count

            df = galpotential(dynamic_components=self.disc_component)
            newpotential=df.potential(R, Z, grid=True, nproc=nproc, toll=inttoll, Rcut=Rcut, zcut=zcut, mcut=mcut,external_potential=fixed_potential,show_comp=False,output='1D')

            tabzd, tabh   = self._calc_flaring(pot_grid=newpotential, vdisp_func=vdisp_func, zlaw=zlaw,outdir=outdir + outfolder, plot=diagnostic, diagnostic=diagnostic,output=diagnostic)
            ftab, fitfunc = self._fit_flaring(tabzd=tabzd, zlaw=flaw, polydegree=polyflare_degree, diagnostic=diagnostic,outdir=outdir + outfolder + '/flare', Rlimit=Rlimit)

            #Stuffs
            residuals         =  np.abs(tabzd - oldtabzd)
            max_residual_abs  =  np.max(residuals)
            max_residual_rel  =  np.max(residuals/tabzd)

            ax10.plot(tabzd[:, 0], tabzd[:, 1],'-o' , color='gray')
            flare_max_tmp=np.max(tabzd[:,1])
            if flare_max_tmp>flare_max: flare_max=flare_max_tmp
            else: pass


            print('Iter-%i: Done'%(count+1))
            sys.stdout.flush()
            print('Max Absolute residual=%.2e'%max_residual_abs)
            print('Max Relative residual=%.2e'%max_residual_rel)

            print('//////////////////////////////////////////////////////////////////////////////\n')
            sys.stdout.flush()

            oldtabzd = tabzd
            count+=1

            fzd=UnivariateSpline(tabzd[:,0],tabzd[:,1],k=2,s=0,ext=3)

        #Output
        ax10.scatter(tabzd[:, 0], tabzd[:, 1], c='blue',s=20,zorder=1000)
        rr=np.linspace(0,R[-1]*1.2,1000)
        if Rlimit is None: y=fitfunc(rr)
        else: y=np.where(rr<=Rlimit,fitfunc(rr),fitfunc(Rlimit))
        ax10.plot(rr,y, '-', c='red')
        ax10.set_xlim(0,rr[-1])
        ax10.set_ylim(0,flare_max*1.05)
        ax10.set_xlabel('R [kpc]',fontsize=20)
        ax10.set_ylabel('Zd [kpc]',fontsize=20)
        fig10.savefig(outdir+'/finalflare_zd.pdf')


        fig11=plt.figure()
        ax11=fig11.add_subplot(111)
        ax11.plot(tabh[:, 0], tabh[:, 1], '-o' , c='blue')
        ax11.set_xlim(0,rr[-1])
        ax11.set_ylim(0,tabh[-1, 1]*1.05)
        ax11.set_xlabel('R [kpc]',fontsize=20)
        ax11.set_ylabel('HWHM [kpc]',fontsize=20)
        fig11.savefig(outdir+'/finalflare_hwhm.pdf')

        #Tab flare
        tabflare=np.zeros(shape=(len(R),3))
        tabflare[:,0]=tabzd[:,0]
        tabflare[:,1]=tabzd[:,1]
        tabflare[:,2]=tabh[:,1]
        np.savetxt(outdir+'/tabflare.dat',tabflare,fmt='%.3f',header='0-R [kpc] 1-Zd [kpc] 2-HWHM [kpc]',footer='Zlaw: %s'%zlaw)

        #Tab Pot
        df_fix.save(filename=outdir+'/tab_fixedpotential.dat', complete=True)
        df.save(filename=outdir+'/tab_totpotential.dat',complete=True)

        return self.disc_component, tabzd, fzd, fitfunc

    def _set_vdisp(self,vdisp):

        if isinstance(vdisp,int) or isinstance(vdisp,float):
            vdisp_func=lambda R: np.where(R==0,vdisp,vdisp)
        elif isinstance(vdisp,float) or isinstance(vdisp,tuple) or isinstance(vdisp,np.ndarray):
            r=vdisp[:,0]
            vd=vdisp[:,1]
            vdisp_func=UnivariateSpline(r,vd,k=2,s=0,ext=3)
        else:
            vdisp_func=vdisp
            try:
                vdisp_func(1)
            except:
                raise ValueError('vdisp_func it not a number a list or a function of one variable')

        return vdisp_func



    def _check_components(self, components):

        if isinstance(components,list) or isinstance(components, tuple) or isinstance(components, np.ndarray):
            i=0
            for comp in components:
                if isinstance(comp, disc) or isinstance(comp, halo):
                    pass
                else:
                    raise ValueError('Dynamic components %i is not from class halo or disc'%i)
                i+=1
        elif isinstance(components, disc) or isinstance(components, halo):
            pass
        else:
            raise ValueError('Dynamic component is not from class halo or disc')

        return 0
