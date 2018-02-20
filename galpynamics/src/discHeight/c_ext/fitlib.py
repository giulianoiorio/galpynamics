from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import time
import datetime
import matplotlib as mpl
import functools as fu
import sys

label_size =18
mpl.rcParams.update({'figure.autolayout':True})
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['mathtext.default']='regular'

#FITTA profili verticali di densita con fitzprof e profili di flaring con fitflare
#V1.1.1

def countrow(tab):
    """
    Conta i valori di R(len(tab)/row) e di Z(row) in una tabella del tipo R-Z-dens
    :param tab:
    :return:
    """
    check=tab[0,0]
    row=np.sum(tab[:,0]==check)
    return int(row),int(len(tab)/row)

def gau(x,sigma,norm=1,mean=0.):
    """
    Gaussian distribution
    :param x: variable
    :param sigma: dispersion
    :param norm: value at x=mean
    :param mean:
    :return:
    """
    return norm*np.exp(-0.5*((x-mean)/sigma)**2)

def sech(x,hs,n=1,norm=1,mean=0.):
    """
    General Hyperbolic secant distribution. Sech(z/zd)^(2/n)
    :param x: Variable
    :param zd: scale length
    :param n: For n=1, sech2 (logistic) distribution, for n=2 sech. For n--->Infty the distribution tends to an Exp
              for n--->0 the distribution tends to a gaussian.
    :param norm: Value for x=mean
    :param mean:
    :return:
    """
    return norm*(1/np.cosh(np.abs((x-mean)/hs))**(2/np.abs(n)))

def exp(x,hs,norm=1,mean=0.):
    """
    Exponential distribution
    :param x: Variable
    :param hs: scale length
    :param norm: Valeu a x=mean
    :param mean:
    :return:
    """
    return norm*(np.exp(-np.abs((x-mean)/hs)))

def lorentz(x,fwhm,back=0,norm=1,mean=0.) :
    """
    Lorentzian distribution
    :param x: Vairable
    :param hwhm: half width half maximum
    :param back: Constant Background
    :param norm: Value at x=mean
    :param mean: Central value
    :return:
    """
    y=(x-mean)/fwhm
    return back+((norm)/(1.0+y**2))

def flaretan(r,h0,rf,c):
    x=r/rf
    return h0+c*np.tanh(x**2)

def flareasin(r,h0,rf,c):
    x=r/rf
    return h0+c*np.arcsinh(x**2)

def fit_bootstrap(z,dens,func,nresample=1000,norm=1,mean=0,weight=None):

    std_results=[]
    lenar=len(dens)
    if weight is None: w=np.array([1,]*lenar)
    else: w=np.array(weight)

    if nresample>0:
        for n in range(nresample):
            indxr=np.random.randint(0,lenar, size=lenar)
            z_tmp=z[indxr]
            dens_tmp=dens[indxr]
            w_tmp=w[indxr]
            par,_= curve_fit(lambda x, s: func(x, s, norm=norm, mean=mean), z_tmp, dens_tmp, p0=(0.2), sigma=w_tmp)
            std_results.append(np.abs(par[0]))

        std_mean=np.nanmean(std_results)
        std_err=np.nanstd(std_results)
    else:

        par, _ = curve_fit(lambda x, s: func(x, s, norm, mean), z, dens, p0=(0.2), sigma=w)

        std_mean = np.abs(par[0])
        std_err=None


    return std_mean,std_err


#TODO: added estimated of errors on zd, but with no output (however the errors ara negiglible)
def fitzprof(zdz,dzlimit=1E-300,nresample=0,output=False,iname='',outdir='fitzprof',Runity='kpc',Zunity='kpc',diagnostic=False,plot=False,**kwargs):
    """
    This function fit variuos functions (gaussian,sech2,sechn,exponential,lorentzian) to a normalized vertical
    distributions.
    For all the functions, we fix the value of the mean value at Z=0 and of the normalizations (ndens(0)=1),
    the only free parameters are the function scale height. The only exception is the Sech(z/zd)^(2/n), in this
    case we fit both the zd and the variable n.
    It return tables with fitted parameters, HWHM and string list of the fuctions used for the fit.
    :param zdz: 3D Table withe col1-Radius, col2-Value of the vertical position Z, col3-Normalized vertical density ndens(Z)=dens(Z)/dens(0).
    :param dzlimit: Limit value of ndensz considered on the fit. This value is added to avoid that the fit
                    try to fit the large value of point with value practically equal to 0.
    :param nresample: If nresample>0, estimate the error with a boostrap method.
    :param output: If true save data and image. default[True]
    :param iname: Identity name of the output, e.g., iname_....  default['']
    :param outdir: Directory where to save the output [fitzprof]
    :param Runity: Unit of the cylindrical radius value. It is a string ['kpc'].
    :param Zunity: Unit of the vertical value. It is a string ['kpc'].
    :param diagnostic: If true plot image and data of the diagnostic analysis of the fits. [True]
    :param plot: If true, save image of the fits, flare and diagnostic. [True]
    :param kwargs:
            dists: Functional forms to use for the fit. The functions allowed
                    are gau-for gaussian, sech2-for hyperbolic squared secant
                    sechn- General hyperbolic secant (sech(z/zd)^(2/n))
                    exp-for exponential  lor-lorentzian.
                    If kwargs not in input, the function use all the allowed functions.
    :return: ftab: Table with the fitted parameters (one per fit functions used except for the 2 of sechn)
             htab: Table with the HWHM of the vertical distribution calculated from the fitted parameters (see master thesis)
             dlist: String list with the name of the used functions.
    """
    nowtime=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    version='1.0'
    print('*'*50)
    print('START FITZPROFILE'.center(50))
    print('*'*50)

    tstart=time.clock()

    #Parameter definition
    Runit=Runity
    Zunit=Zunity



    if output==True:
        if iname!='': iname=iname+'_'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdat=outdir+'/dat'
        if not os.path.exists(outdat):
            os.makedirs(outdat)


    if plot==True:
        outimm=outdir+'/image'
        if not os.path.exists(outimm):
            os.makedirs(outimm)



    i,j=countrow(zdz)
    print('Number of Radii: %i \nNumber of Vertical points: %i' % (j,i))
    #Fixed parameters for the fit
    mean=0.
    norm=1
    back=1
    n=1


    #Check distributions
    alldist=('gau','sech2','sechn','exp','lor') #Allowe distributions
    if 'dists' in kwargs:
        if isinstance(kwargs['dists'],np.ndarray): dlist=kwargs['dists']
        elif isinstance(kwargs['dists'],list): dlist=np.array(kwargs['dists'])
        elif isinstance(kwargs['dists'],tuple): dlist=np.array(kwargs['dists'])
        elif isinstance(kwargs['dists'],str): dlist=np.array([kwargs['dists']])
        else: raise ValueError('dists mut be a list a tuple or a numpy array or a single string')
        #Check if correct distribution
        for kcomp in dlist:
            if kcomp not in alldist: raise ValueError('Allowed distribution are gau,sech,sechn,exp,lor')
        ndist=len(dlist)
    else:
        ndist=5
        dlist=np.array(alldist)

    print('Number of the used distributions: %i'%(ndist),' ',dlist)

    print(ndist)
    if 'sechn' in dlist:
        ftab=np.zeros(shape=(j,ndist+2),dtype=float) #Flare tab
        htab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare hwhm tab
    else:
        ftab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare tab
        htab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare hwhm tab
    if nresample > 0:
        eftab = np.zeros(shape=(j, 2 * ndist + 1), dtype=float)  # Flare  tab with error
        ehtab = np.zeros(shape=(j, 2 * ndist + 1), dtype=float)  # Flare hwhm tab with error

    if diagnostic==True:
        eatab=np.zeros(shape=(j,ndist+1),dtype=float) #Absolute error tab
        ertab=np.zeros(shape=(j,ndist+1),dtype=float) #Relative error tab


    l=0 #Initialize counter

    #Initialize figure



    #Inizialize plot
    if plot==True:
        if diagnostic==True:
            fig=plt.figure(figsize=(8,8)) #Fit plot
            fig2=plt.figure(figsize=(8,8)) #Absolute error plot
            fig3=plt.figure(figsize=(8,8)) #Relative error plot
            nplot=0 #Counting plot
            l=0
            plotrow=4 #Number of row in the plot
            while (nplot<=0):
                nplot=int(round(j/(12-l)))
                l+=1
                if l%3==0: plotrow-=1
            del(l)
            print('nplot',nplot)

    ########################################################################
    #FITTING ROUTINE
    ########################################################################
    cplot=1
    print('---Fitting---')
    for k in range(int(j)):

        #Initialize vectors
        x=zdz[k*i:k*i+i,1]
        z=zdz[k*i:k*i+i,2]
        indfilt=zdz[k*i:k*i+i,2]>dzlimit #Filter value lower than
        x=x[indfilt] #Z values
        z=z[indfilt] #norm dens values
        #weigth=np.where(z>0.1,1/z,0.0001)
        weigth=[1,]*len(x)
        print('Working on radius: %3.2f' % (zdz[k*i,0]))
        if plot==True: xp=np.linspace(0,np.max(x),100)
        #Fit
        if 'gau' in dlist:
            gaupar,egaupar=fit_bootstrap(x, z, gau, nresample=nresample, norm=norm, mean=mean, weight=weigth)
            ygau=gau(x,sigma=gaupar)
            if plot==True: ygaup=gau(xp,sigma=gaupar)
        if 'sech2' in dlist:

            spar,espar=fit_bootstrap(x, z, sech, nresample=nresample, norm=norm, mean=mean, weight=weigth)
            ysec=sech(x,hs=spar)
            if plot==True: ysechp=sech(xp,hs=spar,n=1,mean=0)

        if 'exp' in dlist:

            epar,eepar=fit_bootstrap(x, z, exp, nresample=nresample, norm=norm, mean=mean, weight=weigth)
            yexp=exp(x,hs=epar,norm=1,mean=0)
            if plot == True: yexpp = exp(xp, hs=epar, norm=1, mean=0)

        if 'lor' in dlist:

            lpar,elpar=fit_bootstrap(x, z, lorentz, nresample=nresample, norm=norm, mean=mean, weight=weigth)
            ylor=lorentz(x,fwhm=lpar,back=0,norm=1,mean=0)
            if plot==True: ylorp=lorentz(xp,fwhm=lpar,back=0,norm=1,mean=0)


        #Fill tables
        ftab[k,0]=zdz[k*i,0] #Radii
        htab[k,0]=zdz[k*i,0]
        if nresample>0:
            eftab[k,0]=zdz[k*i,0] #Radii
            ehtab[k,0]=zdz[k*i,0]
        count=1
        cs=0
        for kcomp in dlist:
            if kcomp=='gau':
                ftab[k,count+cs]=np.abs(gaupar)
                htab[k,count]=gaupar*np.sqrt(2*np.log(2))
                if nresample>0:
                    eftab[k, 2*count -1 + cs] = np.abs(gaupar)
                    ehtab[k, 2*count-1] = gaupar * np.sqrt(2 * np.log(2))
                    eftab[k, 2*count  + cs] = np.abs(egaupar)
                    ehtab[k, 2*count] = egaupar * np.sqrt(2 * np.log(2))
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ygau-z))
                    ertab[k,count]=np.sum(np.abs(ygau-z)/z)
            elif kcomp=='sech2':
                ftab[k,count+cs]=np.abs(spar)
                xx=2**(-0.5)
                htab[k,count]=spar*(np.log( (1+np.sqrt(1-xx**2))/xx ))
                if nresample>0:
                    eftab[k, 2*count -1 + cs] = np.abs(spar)
                    ehtab[k, 2*count -1] = spar*(np.log( (1+np.sqrt(1-xx**2))/xx ))
                    eftab[k, 2*count + cs] = np.abs(espar)
                    ehtab[k, 2*count ] = espar*(np.log( (1+np.sqrt(1-xx**2))/xx ))

                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ysec-z))
                    ertab[k,count]=np.sum(np.abs(ysec-z)/z)
            elif kcomp=='exp':
                ftab[k,count+cs]=np.abs(epar)
                htab[k,count]=epar*np.log(2)
                if nresample>0:
                    eftab[k, 2*count -1 + cs] = np.abs(epar)
                    ehtab[k, 2*count -1] = epar*np.log(2)
                    eftab[k, 2*count  + cs] = np.abs(eepar)
                    ehtab[k, 2*count] = epar*np.log(2)
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(yexp-z))
                    ertab[k,count]=np.sum(np.abs(yexp-z)/z)
            elif kcomp=='lor':
                ftab[k,count+cs]=lpar
                htab[k,count]=lpar
                if nresample>0:
                    eftab[k, 2*count -1 + cs] = lpar
                    ehtab[k, 2*count -1] = lpar
                    eftab[k, 2*count  + cs] = elpar
                    ehtab[k, 2*count] = elpar
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ylor-z))
                    ertab[k,count]=np.sum(np.abs(ylor-z)/z)
            count+=1
        del(count)

        if diagnostic==True: #Errors
            eatab[k,0]=zdz[k*i,0]
            ertab[k,0]=zdz[k*i,0]
            #Functions

        if diagnostic==True:
            if plot==True:
                #PLOT
                w=(k+1)%nplot #checkplot
                if cplot<=12:
                    if w==0:
                        print('Plotting')
                        #Fit plot
                        ax=fig.add_subplot(plotrow,3,cplot)
                        if cplot==1:
                            legendlist=[]
                            if 'gau' in dlist:
                                gaup,=ax.plot(xp,ygaup,c='red',lw=0.5,label='gau')
                                legendlist.append(gaup)
                            if 'sech2' in dlist:
                                sech2p,=ax.plot(xp,ysechp,c='green',lw=0.5,label='sech2')
                                legendlist.append(sech2p)
                            if 'sechn' in dlist:
                                sechnp,=ax.plot(xp,ysecnp,c='orange',lw=0.5,label='sechn')
                                legendlist.append(sechnp)
                            if 'exp' in dlist:
                                ep,=ax.plot(xp,yexpp,c='black',lw=0.5,label='exp')
                                legendlist.append(ep)
                            if 'lor' in dlist:
                                lp,=ax.plot(xp,ylorp,c='magenta',lw=0.5,label='lor')
                                legendlist.append(lp)
                        else:
                            if 'gau' in dlist: ax.plot(xp,ygaup,c='red',lw=0.5,label='gau')
                            if 'sech2' in dlist: ax.plot(xp,ysechp,c='green',lw=0.5,label='sech2')
                            if 'sechn' in dlist: ax.plot(xp,ysecnp,c='orange',lw=0.5,label='sechn')
                            if 'exp' in dlist: ax.plot(xp,yexpp,c='black',lw=0.5,label='exp')
                            if 'lor' in dlist:
                                ax.plot(xp,ylorp,c='magenta',lw=0.5,label='lor')


                        ax.scatter(x,z,s=2)
                        ax.tick_params(axis='x', labelsize=5)
                        ax.tick_params(axis='y', labelsize=5)
                        ax.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax.set_ylabel(r'$\tilde{\rho}$',fontsize=10)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax.set_xlabel('Z [%s]'%(Zunit),fontsize=8)


                        #Rel_err plot
                        ax2=fig2.add_subplot(plotrow,3,cplot)
                        if 'gau' in dlist: ax2.plot(x,np.abs(ygau-z)/z,c='red',lw=0.5,label='gau')
                        if 'sech2' in dlist: ax2.plot(x,np.abs(ysec-z)/z,c='green',lw=0.5,label='sech2')
                        if 'sechn' in dlist: ax2.plot(x,np.abs(ysecn-z)/z,c='orange',lw=0.5,label='sechn')
                        if 'exp' in dlist: ax2.plot(x,np.abs(yexp-z)/z,c='black',lw=0.5,label='exp')
                        if 'lor' in dlist: ax2.plot(x,np.abs(ylor-z)/z,c='magenta',lw=0.5,label='lor')
                        ax2.tick_params(axis='x', labelsize=5)
                        ax2.tick_params(axis='y', labelsize=5)
                        ax2.set_ylim(0,1)
                        ax2.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax2.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax2.set_ylabel('Rel. err',fontsize=8)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax2.set_xlabel('Z [%s]'%(Zunit),fontsize=8)


                        #Abs_err plot
                        ax3=fig3.add_subplot(plotrow,3,cplot)
                        if 'gau' in dlist: ax3.plot(x,np.abs(ygau-z),c='red',lw=0.5,label='gau')
                        if 'sech2' in dlist: ax3.plot(x,np.abs(ysec-z),c='green',lw=0.5,label='sech2')
                        if 'sechn' in dlist: ax3.plot(x,np.abs(ysecn-z),c='orange',lw=0.5,label='sechn')
                        if 'exp' in dlist: ax3.plot(x,np.abs(yexp-z),c='black',lw=0.5,label='exp')
                        if 'lor' in dlist: ax3.plot(x,np.abs(ylor-z),c='magenta',lw=0.5,label='lor')
                        ax3.tick_params(axis='x', labelsize=5)
                        ax3.tick_params(axis='y', labelsize=5)
                        ax3.set_ylim(0,0.1)
                        ax3.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax3.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax3.set_ylabel('Abs. err',fontsize=8)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax3.set_xlabel('Z [%s]'%(Zunit),fontsize=8)

                        cplot+=1


    #PLOTTING TO FILE

    if plot==True:
        print('Save figures')
        sys.stdout.flush()
        if diagnostic==True:
            fig.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)
            fig2.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)
            fig3.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)

            fig.savefig(outimm+'/'+iname+'fit.pdf')
            fig2.savefig(outimm+'/'+iname+'rel_err.pdf')
            fig3.savefig(outimm+'/'+iname+'abs_err.pdf')

            #plt absolute error
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            kk=1
            lcol=('red','green','orange','black','magenta')
            for kcomp in dlist:
                ax.scatter(eatab[:,0],eatab[:,kk],c=lcol[kk-1],lw=0.5,label=kcomp)
                kk+=1
            plt.legend(fontsize=13,loc='upper center',bbox_to_anchor=[0.5,1.1],ncol=ndist)
            ax.set_xlabel('R [%s]' %(Runit))
            ax.set_ylabel('Cumulative Abs. err')
            plt.savefig(outimm+'/'+iname+'abserrtot.pdf')
            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)

        #Plt flare

        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        kk=1
        lcol=('red','green','orange','black','magenta')
        for kcomp in dlist:

            vall=htab[:,kk]
            if nresample>0: evall=ehtab[:,2*kk]
            else: evall=None

            ax.errorbar(htab[:,0],vall,evall,fmt='-o',color=lcol[kk-1],lw=1,label=kcomp)
            kk+=1
        ax.legend(loc='upper left')
        ax.set_xlabel('R [%s]' %(Runit))
        ax.set_ylabel('HWHM [%s]' %(Zunit))
        plt.legend(loc='upper left',ncol=ndist)
        plt.legend(fontsize=13,loc='upper center',bbox_to_anchor=[0.5,1.1],ncol=ndist)
        plt.savefig(outimm+'/'+iname+'flare.pdf')
        plt.close(fig)

    if output==True:
        #TABLE
        print('Writing table')
        #Create a header for the ouputted  file

        if 'sechn' in dlist: lcomp= ' '*9+'|'+' '+'Function Scale Height'.center(10*(len(dlist)+1))
        else: lcomp=' '*9+'|'+' '+'Function Scale Height (Hs)'.center(10*len(dlist))

        lcomp+='\n'+'R'.center(10) + ' '
        for kcomp in dlist:
            if kcomp=='sechn':
                tmp=kcomp+'(hs)'
                lcomp+=tmp.center(10) + ' '
                tmp=kcomp+'(n)'
                lcomp+=tmp.center(10) + ' '
            else: lcomp+= kcomp.center(10) + ' '

        comf='Fitted height scale for various distributions. In the case of the Sechn, it is printed also ' \
             'the value of the factor n.'
        comf+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)
        comf=comf + '\n' + 'Length units: R - '+Runit + ', ' + 'Hs - ' + Zunit + '\n' + lcomp


        comh='HWHM for the various distributions.'
        comh+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)
        comh+='\n' + 'File made with FitFlare %s on: %s ' % (version,nowtime)
        comh=comh + '\n' + 'Length units: R - '+Runit + ', ' + 'HWHM - ' + Zunit
        comh+='\n'+' '*9+'|'+' '+'HWHM'.center(10*len(dlist))

        lcomp='R'.center(10) + ' '
        for kcomp in dlist:
            lcomp+= kcomp.center(10) + ' '

        comh+='\n' + lcomp
        #WRITE TABLE DAT
        np.savetxt(outdir+'/dat/'+iname+'flare.dat',ftab,header=comf,fmt='%10.4f')
        np.savetxt(outdir+'/dat/'+iname+'flare_hwhm.dat',htab,header=comh,fmt='%10.4f')
        del(comf)
        del(comh)

        if diagnostic==True:
            commerr='Cumulative absolute error for the various distributions.'
            commerr+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)

            lcomp=' '*9+'|'+' '+'Cumulative absolute error'.center(15*len(dlist)) +'\n'
            lcomp+='R'.center(15) + ' '
            for kcomp in dlist:
                lcomp+= kcomp.center(15) + ' '

            commerr=commerr + '\n' + 'Length units: R - '+Runit  + '\n'  +  lcomp
            np.savetxt(outdir+'/dat/'+iname+'fit_abserr.dat',htab,header=commerr,fmt='%15.4e')

    #plt.close()
    dt=time.clock()-tstart
    print('DONE in %.3f minutes' % (dt/60) )
    if output == True: print('Output  data files in %s' % (outdir+'/dat'))
    if plot==True: print('Output  images in %s' % (outdir+'/image'))

    print('*'*50)
    print('END FITZPROFILE'.center(50))
    print('*'*50)

    return(ftab,htab,dlist)

'''
Old Backup
#TODO: add the bootstrap estimate of errors
def fitzprof(zdz,dzlimit=1E-300,output=False,iname='',outdir='fitzprof',Runity='kpc',Zunity='kpc',diagnostic=False,plot=False,**kwargs):
    """
    This function fit variuos functions (gaussian,sech2,sechn,exponential,lorentzian) to a normalized vertical
    distributions.
    For all the functions, we fix the value of the mean value at Z=0 and of the normalizations (ndens(0)=1),
    the only free parameters are the function scale height. The only exception is the Sech(z/zd)^(2/n), in this
    case we fit both the zd and the variable n.
    It return tables with fitted parameters, HWHM and string list of the fuctions used for the fit.
    :param zdz: 3D Table withe col1-Radius, col2-Value of the vertical position Z, col3-Normalized vertical density ndens(Z)=dens(Z)/dens(0).
    :param dzlimit: Limit value of ndensz considered on the fit. This value is added to avoid that the fit
                    try to fit the large value of point with value practically equal to 0.
    :param output: If true save data and image. default[True]
    :param iname: Identity name of the output, e.g., iname_....  default['']
    :param outdir: Directory where to save the output [fitzprof]
    :param Runity: Unit of the cylindrical radius value. It is a string ['kpc'].
    :param Zunity: Unit of the vertical value. It is a string ['kpc'].
    :param diagnostic: If true plot image and data of the diagnostic analysis of the fits. [True]
    :param plot: If true, save image of the fits, flare and diagnostic. [True]
    :param kwargs:
            dists: Functional forms to use for the fit. The functions allowed
                    are gau-for gaussian, sech2-for hyperbolic squared secant
                    sechn- General hyperbolic secant (sech(z/zd)^(2/n))
                    exp-for exponential  lor-lorentzian.
                    If kwargs not in input, the function use all the allowed functions.
    :return: ftab: Table with the fitted parameters (one per fit functions used except for the 2 of sechn)
             htab: Table with the HWHM of the vertical distribution calculated from the fitted parameters (see master thesis)
             dlist: String list with the name of the used functions.
    """
    nowtime=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    version='1.0'
    print('*'*50)
    print('START FITZPROFILE'.center(50))
    print('*'*50)

    tstart=time.clock()

    #Parameter definition
    Runit=Runity
    Zunit=Zunity



    if output==True:
        if iname!='': iname=iname+'_'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdat=outdir+'/dat'
        if not os.path.exists(outdat):
            os.makedirs(outdat)


    if plot==True:
        outimm=outdir+'/image'
        if not os.path.exists(outimm):
            os.makedirs(outimm)



    i,j=countrow(zdz)
    print('Number of Radii: %i \nNumber of Vertical points: %i' % (j,i))
    #Fixed parameters for the fit
    mean=0.
    norm=1
    back=1
    n=1


    #Check distributions
    alldist=('gau','sech2','sechn','exp','lor') #Allowe distributions
    if 'dists' in kwargs:
        if isinstance(kwargs['dists'],np.ndarray): dlist=kwargs['dists']
        elif isinstance(kwargs['dists'],list): dlist=np.array(kwargs['dists'])
        elif isinstance(kwargs['dists'],tuple): dlist=np.array(kwargs['dists'])
        elif isinstance(kwargs['dists'],str): dlist=np.array([kwargs['dists']])
        else: raise ValueError('dists mut be a list a tuple or a numpy array or a single string')
        #Check if correct distribution
        for kcomp in dlist:
            if kcomp not in alldist: raise ValueError('Allowed distribution are gau,sech,sechn,exp,lor')
        ndist=len(dlist)
    else:
        ndist=5
        dlist=np.array(alldist)

    print('Number of the used distributions: %i'%(ndist),' ',dlist)

    print(ndist)
    if 'sechn' in dlist:
        ftab=np.zeros(shape=(j,ndist+2),dtype=float) #Flare tab
        htab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare hwhm tab
    else:
        ftab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare tab
        htab=np.zeros(shape=(j,ndist+1),dtype=float) #Flare hwhm tab
    if diagnostic==True:
        eatab=np.zeros(shape=(j,ndist+1),dtype=float) #Absolute error tab
        ertab=np.zeros(shape=(j,ndist+1),dtype=float) #Relative error tab


    l=0 #Initialize counter

    #Initialize figure



    #Inizialize plot
    if plot==True:
        if diagnostic==True:
            fig=plt.figure(figsize=(8,8)) #Fit plot
            fig2=plt.figure(figsize=(8,8)) #Absolute error plot
            fig3=plt.figure(figsize=(8,8)) #Relative error plot
            nplot=0 #Counting plot
            l=0
            plotrow=4 #Number of row in the plot
            while (nplot<=0):
                nplot=int(round(j/(12-l)))
                l+=1
                if l%3==0: plotrow-=1
            del(l)
            print('nplot',nplot)

    ########################################################################
    #FITTING ROUTINE
    ########################################################################
    cplot=1
    print('---Fitting---')
    for k in range(int(j)):

        #Initialize vectors
        x=zdz[k*i:k*i+i,1]
        z=zdz[k*i:k*i+i,2]
        indfilt=zdz[k*i:k*i+i,2]>dzlimit #Filter value lower than
        x=x[indfilt] #Z values
        z=z[indfilt] #norm dens values
        #weigth=np.where(z>0.1,1/z,0.0001)
        weigth=[1,]*len(x)
        print('Working on radius: %3.2f' % (zdz[k*i,0]))
        if plot==True: xp=np.linspace(0,np.max(x),100)
        #Fit
        if 'gau' in dlist:
            gaupar,_=curve_fit(lambda x,s: gau(x,s,norm=norm,mean=mean),x,z,p0=(0.2),sigma=weigth)
            gaupar=np.abs(gaupar)
            ygau=gau(x,sigma=gaupar[0])
            if plot==True: ygaup=gau(xp,sigma=gaupar[0])
        if 'sech2' in dlist:
            spar,_=curve_fit(lambda x,s: sech(x,s,norm=norm,mean=mean,n=1),x,z,sigma=weigth)
            spar=np.abs(spar)
            ysec=sech(x,hs=spar[0],n=1,mean=0)
            if plot==True: ysechp=sech(xp,hs=spar[0],n=1,mean=0)
        if 'sechn' in dlist:
            #In some case the sech^(1/n) do not converge (few points), in this case
            #we use the parameters found for the classical sech^2 (spar)
            try:
                snpar,_=curve_fit(lambda x,s,m: sech(x,s,m,norm,mean),x,z,sigma=weigth)
                snpar=np.abs(snpar)
                ysecn=sech(x,hs=snpar[0],n=snpar[1],mean=0)
                if plot==True: ysecnp=sech(xp,hs=snpar[0],n=snpar[1],mean=0)
            except:
                spar,_=curve_fit(lambda x,s: sech(x,s,n,norm,mean),x,z,sigma=weigth)
                snpar=[spar[0],1]
                snpar=np.abs(snpar)
                ysecn=sech(x,hs=snpar[0],n=snpar[1],mean=0)
                if plot==True: ysecnp=sech(xp,hs=spar[0],n=1,mean=0)
        if 'exp' in dlist:
            epar,_=curve_fit(lambda x,s: exp(x,s,norm,mean),x,z,sigma=weigth)
            epar=np.abs(epar)
            yexp=exp(x,hs=epar[0],norm=1,mean=0)
            if plot==True: yexpp=exp(xp,hs=epar[0],norm=1,mean=0)
        if 'lor' in dlist:
            lpar,_=curve_fit(lambda x,s: lorentz(x,s,norm,mean),x,z,sigma=weigth)
            lpar=np.abs(lpar)
            ylor=lorentz(x,fwhm=lpar[0],back=0,norm=1,mean=0)
            if plot==True: ylorp=lorentz(xp,fwhm=lpar[0],back=0,norm=1,mean=0)


        #Fill tables
        ftab[k,0]=zdz[k*i,0] #Radii
        htab[k,0]=zdz[k*i,0]
        count=1
        cs=0
        for kcomp in dlist:
            if kcomp=='gau':
                ftab[k,count+cs]=np.abs(gaupar[0])
                htab[k,count]=gaupar[0]*np.sqrt(2*np.log(2))
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ygau-z))
                    ertab[k,count]=np.sum(np.abs(ygau-z)/z)
            elif kcomp=='sech2':
                ftab[k,count+cs]=np.abs(spar[0])
                xx=2**(-0.5)
                htab[k,count]=spar[0]*(np.log( (1+np.sqrt(1-xx**2))/xx ))
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ysec-z))
                    ertab[k,count]=np.sum(np.abs(ysec-z)/z)
            elif kcomp=='sechn':
                ftab[k,count]=np.abs(snpar[0])
                ftab[k,count+1]=np.abs(snpar[1])
                xx=2**(-snpar[1]/2)
                htab[k,count]=snpar[0]*(np.log( (1+np.sqrt(1-xx**2))/xx ))
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ysecn-z))
                    ertab[k,count]=np.sum(np.abs(ysecn-z)/z)
                cs=1
            elif kcomp=='exp':
                ftab[k,count+cs]=np.abs(epar[0])
                htab[k,count]=epar[0]*np.log(2)
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(yexp-z))
                    ertab[k,count]=np.sum(np.abs(yexp-z)/z)
            elif kcomp=='lor':
                ftab[k,count+cs]=lpar[0]
                htab[k,count]=lpar[0]
                if diagnostic==True:
                    eatab[k,count]=np.sum(np.abs(ylor-z))
                    ertab[k,count]=np.sum(np.abs(ylor-z)/z)
            count+=1
        del(count)

        if diagnostic==True: #Errors
            eatab[k,0]=zdz[k*i,0]
            ertab[k,0]=zdz[k*i,0]
            #Functions

        if diagnostic==True:
            if plot==True:
                #PLOT
                w=(k+1)%nplot #checkplot
                if cplot<=12:
                    if w==0:
                        print('Plotting')
                        #Fit plot
                        ax=fig.add_subplot(plotrow,3,cplot)
                        if cplot==1:
                            legendlist=[]
                            if 'gau' in dlist:
                                gaup,=ax.plot(xp,ygaup,c='red',lw=0.5,label='gau')
                                legendlist.append(gaup)
                            if 'sech2' in dlist:
                                sech2p,=ax.plot(xp,ysechp,c='green',lw=0.5,label='sech2')
                                legendlist.append(sech2p)
                            if 'sechn' in dlist:
                                sechnp,=ax.plot(xp,ysecnp,c='orange',lw=0.5,label='sechn')
                                legendlist.append(sechnp)
                            if 'exp' in dlist:
                                ep,=ax.plot(xp,yexpp,c='black',lw=0.5,label='exp')
                                legendlist.append(ep)
                            if 'lor' in dlist:
                                lp,=ax.plot(xp,ylorp,c='magenta',lw=0.5,label='lor')
                                legendlist.append(lp)
                        else:
                            if 'gau' in dlist: ax.plot(xp,ygaup,c='red',lw=0.5,label='gau')
                            if 'sech2' in dlist: ax.plot(xp,ysechp,c='green',lw=0.5,label='sech2')
                            if 'sechn' in dlist: ax.plot(xp,ysecnp,c='orange',lw=0.5,label='sechn')
                            if 'exp' in dlist: ax.plot(xp,yexpp,c='black',lw=0.5,label='exp')
                            if 'lor' in dlist:
                                ax.plot(xp,ylorp,c='magenta',lw=0.5,label='lor')


                        ax.scatter(x,z,s=2)
                        ax.tick_params(axis='x', labelsize=5)
                        ax.tick_params(axis='y', labelsize=5)
                        ax.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax.set_ylabel(r'$\tilde{\rho}$',fontsize=10)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax.set_xlabel('Z [%s]'%(Zunit),fontsize=8)


                        #Rel_err plot
                        ax2=fig2.add_subplot(plotrow,3,cplot)
                        if 'gau' in dlist: ax2.plot(x,np.abs(ygau-z)/z,c='red',lw=0.5,label='gau')
                        if 'sech2' in dlist: ax2.plot(x,np.abs(ysec-z)/z,c='green',lw=0.5,label='sech2')
                        if 'sechn' in dlist: ax2.plot(x,np.abs(ysecn-z)/z,c='orange',lw=0.5,label='sechn')
                        if 'exp' in dlist: ax2.plot(x,np.abs(yexp-z)/z,c='black',lw=0.5,label='exp')
                        if 'lor' in dlist: ax2.plot(x,np.abs(ylor-z)/z,c='magenta',lw=0.5,label='lor')
                        ax2.tick_params(axis='x', labelsize=5)
                        ax2.tick_params(axis='y', labelsize=5)
                        ax2.set_ylim(0,1)
                        ax2.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax2.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax2.set_ylabel('Rel. err',fontsize=8)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax2.set_xlabel('Z [%s]'%(Zunit),fontsize=8)


                        #Abs_err plot
                        ax3=fig3.add_subplot(plotrow,3,cplot)
                        if 'gau' in dlist: ax3.plot(x,np.abs(ygau-z),c='red',lw=0.5,label='gau')
                        if 'sech2' in dlist: ax3.plot(x,np.abs(ysec-z),c='green',lw=0.5,label='sech2')
                        if 'sechn' in dlist: ax3.plot(x,np.abs(ysecn-z),c='orange',lw=0.5,label='sechn')
                        if 'exp' in dlist: ax3.plot(x,np.abs(yexp-z),c='black',lw=0.5,label='exp')
                        if 'lor' in dlist: ax3.plot(x,np.abs(ylor-z),c='magenta',lw=0.5,label='lor')
                        ax3.tick_params(axis='x', labelsize=5)
                        ax3.tick_params(axis='y', labelsize=5)
                        ax3.set_ylim(0,0.1)
                        ax3.text(.5,1.03,'R= %3.2f %s' % (zdz[k*i,0],Runit),horizontalalignment='center',fontsize=6,transform=ax3.transAxes)
                        if ((cplot==1) | (cplot==4) | (cplot==7) | (cplot==10) ):
                            ax3.set_ylabel('Abs. err',fontsize=8)
                        if ((cplot==plotrow*3-2) | (cplot==plotrow*3-1) | (cplot==plotrow*3)  ):
                            ax3.set_xlabel('Z [%s]'%(Zunit),fontsize=8)

                        cplot+=1


    #PLOTTING TO FILE

    if plot==True:
        print('Save figures')
        sys.stdout.flush()
        if diagnostic==True:
            fig.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)
            fig2.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)
            fig3.legend(legendlist,dlist,fontsize=8,loc='upper center',bbox_to_anchor=[0.5,0.97],ncol=5)

            fig.savefig(outimm+'/'+iname+'fit.pdf')
            fig2.savefig(outimm+'/'+iname+'rel_err.pdf')
            fig3.savefig(outimm+'/'+iname+'abs_err.pdf')

            #plt absolute error
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            kk=1
            lcol=('red','green','orange','black','magenta')
            for kcomp in dlist:
                ax.scatter(eatab[:,0],eatab[:,kk],c=lcol[kk-1],lw=0.5,label=kcomp)
                kk+=1
            plt.legend(fontsize=13,loc='upper center',bbox_to_anchor=[0.5,1.1],ncol=ndist)
            ax.set_xlabel('R [%s]' %(Runit))
            ax.set_ylabel('Cumulative Abs. err')
            plt.savefig(outimm+'/'+iname+'abserrtot.pdf')
            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)

        #Plt flare

        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        kk=1
        lcol=('red','green','orange','black','magenta')
        for kcomp in dlist:
            ax.plot(htab[:,0],htab[:,kk],c=lcol[kk-1],lw=0.5,label=kcomp)
            kk+=1
        ax.legend(loc='upper left')
        ax.set_xlabel('R [%s]' %(Runit))
        ax.set_ylabel('HWHM [%s]' %(Zunit))
        plt.legend(loc='upper left',ncol=ndist)
        plt.legend(fontsize=13,loc='upper center',bbox_to_anchor=[0.5,1.1],ncol=ndist)
        plt.savefig(outimm+'/'+iname+'flare.pdf')
        plt.close(fig)

    if output==True:
        #TABLE
        print('Writing table')
        #Create a header for the ouputted  file

        if 'sechn' in dlist: lcomp= ' '*9+'|'+' '+'Function Scale Height'.center(10*(len(dlist)+1))
        else: lcomp=' '*9+'|'+' '+'Function Scale Height (Hs)'.center(10*len(dlist))

        lcomp+='\n'+'R'.center(10) + ' '
        for kcomp in dlist:
            if kcomp=='sechn':
                tmp=kcomp+'(hs)'
                lcomp+=tmp.center(10) + ' '
                tmp=kcomp+'(n)'
                lcomp+=tmp.center(10) + ' '
            else: lcomp+= kcomp.center(10) + ' '

        comf='Fitted height scale for various distributions. In the case of the Sechn, it is printed also ' \
             'the value of the factor n.'
        comf+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)
        comf=comf + '\n' + 'Length units: R - '+Runit + ', ' + 'Hs - ' + Zunit + '\n' + lcomp


        comh='HWHM for the various distributions.'
        comh+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)
        comh+='\n' + 'File made with FitFlare %s on: %s ' % (version,nowtime)
        comh=comh + '\n' + 'Length units: R - '+Runit + ', ' + 'HWHM - ' + Zunit
        comh+='\n'+' '*9+'|'+' '+'HWHM'.center(10*len(dlist))

        lcomp='R'.center(10) + ' '
        for kcomp in dlist:
            lcomp+= kcomp.center(10) + ' '

        comh+='\n' + lcomp
        #WRITE TABLE DAT
        np.savetxt(outdir+'/dat/'+iname+'flare.dat',ftab,header=comf,fmt='%10.4f')
        np.savetxt(outdir+'/dat/'+iname+'flare_hwhm.dat',htab,header=comh,fmt='%10.4f')
        del(comf)
        del(comh)

        if diagnostic==True:
            commerr='Cumulative absolute error for the various distributions.'
            commerr+='\n' + 'File made with FitzProfile %s on: %s ' % (version,nowtime)

            lcomp=' '*9+'|'+' '+'Cumulative absolute error'.center(15*len(dlist)) +'\n'
            lcomp+='R'.center(15) + ' '
            for kcomp in dlist:
                lcomp+= kcomp.center(15) + ' '

            commerr=commerr + '\n' + 'Length units: R - '+Runit  + '\n'  +  lcomp
            np.savetxt(outdir+'/dat/'+iname+'fit_abserr.dat',htab,header=commerr,fmt='%15.4e')

    #plt.close()
    dt=time.clock()-tstart
    print('DONE in %.3f minutes' % (dt/60) )
    if output == True: print('Output  data files in %s' % (outdir+'/dat'))
    if plot==True: print('Output  images in %s' % (outdir+'/image'))

    print('*'*50)
    print('END FITZPROFILE'.center(50))
    print('*'*50)

    return(ftab,htab,dlist)
'''

def fitflare(tab,func='poly',polydegree=4,outfile=True,plot=True,outdir='fitflare',iname='',Rlimit=None,**kwargs):
    """

    :param tab: Tabella con i valori del flare, deve avere almeno due colonne e la prima colonna deve contenere il raggio.
    :param func: Forma funzionale da usare nel fit:
                tanh o t per usare la forma  h0 + c*Tan((r/rf)^2)
                asinh o a per usare la forma h0 + c*Arcsinh((r/rf)^2)
                both o b per usare tutte e due.
                [both]
    :param outfile: Se True salva i file con i risultati del plot [True]
    :param plot: Se True salva l'immagine con i dati del flare piu i fit [True]
    :param outdir: Nome della directory dove salvare gli output [fitflare]
    :param iname: Nome identificativo per i file in output (E.g. le immagini salvate saranno chiamate iname_....) default['']
    :param kwargs: Possibili opzioni
                comp= Se questa keyword e inserita, va fornita una lista di stringhe contenti identificativi dei vari flare
                passati al programma
    :return: Un np.array contenente per ogni riga i parametri fittati delle forme funzionali scelte per i componenti dati in output.
            in piu viene fornita anche una stima dell errore sulla 4 e eventualmente 8 colonna, calcolato come
            sum ( abs(fit-data) )/Ndata. Parameters with values 0,0,1,inf indicate a failure of the fit routine.
    """


    print('*'*50)
    print('START FITFLARE'.center(50))
    print('*'*50)
    version='1.0'


    if (outfile==True) | (plot==True):
        if iname!='': iname+='_'
        savename=outdir+'/'+iname
        if not os.path.exists(outdir):
            os.makedirs(outdir)


    if ((func=='tanh') | (func=='t')):
        col=4
    elif ((func=='asinh') | (func=='a')):
        col=4
    elif ((func=='poly') | (func=='p')):
        col=polydegree+2
    else: raise NotImplementedError('Flarelaw %s not implemented yet'%func)


    print('Start fitting')

    ftab=np.zeros(shape=(col),dtype=float)

    r=tab[:,0]
    zd=tab[:,1]

    if ((func=='tanh') | (func=='t')):

        p0=(zd[0],np.mean(r),1)
        tpar,_ = curve_fit(flaretan,r,zd,p0=p0)
        tres=np.sum(np.abs(zd-flaretan(r,tpar[0],tpar[1],tpar[2])))/(len(r))
        ftab[0:3]=tpar
        ftab[-1]=tres

    elif ((func=='asinh') | (func=='a')):

        p0 = (zd[0], np.mean(r), 1)
        tpar, _ = curve_fit(flaretan, r, zd, p0=p0)
        tres = np.sum(np.abs(zd - flareasin(r, tpar[0], tpar[1], tpar[2]))) / (len(r))
        ftab[0:3] = tpar
        ftab[-1] = tres

    elif ((func=='poly') | (func=='p')):

        coftab=np.polyfit(r,zd,deg=polydegree)
        pol=np.poly1d(coftab)
        coftab=coftab[::-1]
        tres=np.sum(np.abs(zd - pol(r))) / (len(r))

        ftab[:-1] = coftab
        ftab[-1]  = tres


    if outfile==True:
        print('Writing table')
        if (func=='a') | (func=='asinh'):
            func2='h0 + c*Arcsinh((r/rf)^2)'
        elif (func=='t') | (func=='tanh'):
            func2='h0 + c*Tan((r/rf)^2)'
        elif (func=='p') | (func=='poly'):
            func2='Poly degree=%s'%polydegree

        head=''
        head+='\n'+'Fitting functions: %s' % (func2)

        print('Save table')
        np.savetxt(savename+'fitflare_par.dat',ftab,fmt='%8.4f',header=head)

    if plot==True:
        print('Make plot')
        rr=np.linspace(0.,r[-1]*1.5,100)
        if Rlimit is None: Rlimit=rr[-1]
        fig=plt.figure(figsize=(8,8))
        axr=1
        axc=1


        ax=fig.add_subplot(axr,axc,1)
        ax.set_xlim(-0.1,rr[-1])
        ax.set_xlabel('R [kpc]',fontsize=20)
        ax.set_ylabel('Zd [kpc]',fontsize=20)

        ax.scatter(r,zd,label='Flaring',c='black',s=40,zorder=1000)
        if (func=='tanh') | (func=='t'):
            h0=ftab[0]
            Rf=ftab[1]
            c=ftab[2]
            y=np.where(rr<=Rlimit,flaretan(rr,h0,Rf,c), flaretan(Rlimit,h0,Rf,c))
            ax.plot(rr,y,c='r',label='Tanh')
            fitfunc=fu.partial(flaretan,h0=h0, rf=Rf, c=c)
        elif (func=='asinh') | (func=='a'):
            h0=ftab[0]
            Rf=ftab[1]
            c=ftab[2]
            y=np.where(rr<=Rlimit,flareasin(rr,h0,Rf,c), flareasin(Rlimit,h0,Rf,c))
            ax.plot(rr,y,c='blue',label='Arcsinh')
            fitfunc=fu.partial(flareqsin,h0=h0, rf=Rf, c=c)
        elif ((func == 'poly') | (func == 'pol')) | (func=='p'):
            coeff=ftab[:-1]
            p=np.poly1d(coeff[::-1])
            y=np.where(rr<=Rlimit,p(rr),p(Rlimit))
            ax.plot(rr,y,c='darkgreen',label='Polynomial %ith degree'%polydegree)
            fitfunc=p
        else:
            raise ValueError()
        ax.set_ylim(0., np.max(y)*1.05)


        print('Save plot')

        ax.legend(loc='upper left',fontsize=13,ncol=3)
        fig.subplots_adjust(hspace=0.,wspace=0.)
        fig.savefig(savename+'flare.pdf')

    if outfile==True: print('data in %s' % (savename+'fitflare_par.dat') )
    if plot==True: print('image in %s' % (savename+'flare.pdf') )

    print('*'*50)
    print('END FITFLARE'.center(50))
    print('*'*50)

    return ftab, fitfunc


