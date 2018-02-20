from __future__ import division, print_function
from ..pot_disc.pot_disc import disc
from ..pot_halo.pot_halo import  halo
import numpy as np
import time
import copy
import sys

#TODO: change name in galmodel and reogarnize
class galpotential:


    def __init__(self,dynamic_components=()):

        self._check_components(dynamic_components)
        if isinstance(dynamic_components,list) or isinstance(dynamic_components,tuple) or isinstance(dynamic_components,np.ndarray):
            self.dynamic_components=list(dynamic_components)
            self.ncomp=len(self.dynamic_components)
        else:
            self.dynamic_components=(dynamic_components,)
            self.ncomp=1
        self.potential_grid=None
        self.external_potential=None
        self.potential_grid_exist=False

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

    def add_components(self,components=()):

        self._check_components(components)

        self.dynamic_components=self.dynamic_components+list(components)

        self.ncomp=len(self.dynamic_components)

        return 0

    def remove_components(self,idx=()):

        dynamic_components=[]

        for i in range(len(self.dynamic_components)):

            for j in idx:
                if i!=j:
                    print('i',i,idx)
                    dynamic_components.append(self.dynamic_components[i])
                else:
                    pass

        self.dynamic_components=dynamic_components

        self.ncomp = len(self.dynamic_components)

        return 0

    def _make_finalgrid(self,R,Z,ncolumn=3,grid=False):

        if isinstance(R,float) or isinstance(R, int): R=np.array((R,))
        if isinstance(Z,float) or isinstance(Z, int): Z=np.array((Z,))


        lenR=len(R)
        lenZ=len(Z)


        if lenR!=lenZ or grid==True:
            nrow=lenR*lenZ
        else:
            nrow=lenR

        arr=np.zeros(shape=(nrow,ncolumn))

        return arr

    def potential(self,R,Z,grid=False,nproc=2, toll=1e-4, Rcut=None, zcut=None, mcut=None,external_potential=None, output='1D',show_comp=True):


        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for disc.potential'%str(output))


        grid_final=self._make_finalgrid(R,Z,ncolumn=3,grid=grid)
        grid_complete=self._make_finalgrid(R,Z,ncolumn=len(self.dynamic_components)+4,grid=grid)
        self.external_potential=external_potential

        #External potential
        print('External potential: ',end='')
        sys.stdout.flush()
        if external_potential is not None:
            if len(external_potential)!=len(grid_final):
                raise ValueError('External potential dimension (%i) are than the user defined grid dimension (%i)'%(len(external_potential),len(grid_final)))
            else:
                grid_complete[:,-2]=external_potential[:,-1]
                grid_final[:,-1]=external_potential[:,-1]
                sys.stdout.flush()
        else:
            sys.stdout.flush()

        #Calc potential
        i=0
        for comp in self.dynamic_components:
            print('Calculating Potential of the %ith component (%s)...'%(i+1,comp.name),end='')
            sys.stdout.flush()
            if isinstance(comp, halo):
                tini=time.time()
                grid_tmp = comp.potential(R, Z, grid=grid, toll=toll, mcut=mcut, nproc=nproc,output='1D')
                tfin=time.time()
            elif isinstance(comp,disc):
                tini=time.time()
                grid_tmp = comp.potential(R,Z,grid=grid,toll=toll,Rcut=Rcut, zcut=zcut, nproc=nproc,output='1D')
                tfin=time.time()
            tottime=tfin-tini
            print('Done (%.2f s)'%tottime)
            if i==0:
                grid_final[:,0]=grid_tmp[:,0]
                grid_final[:,1]=grid_tmp[:,1]
                grid_final[:,2]+=grid_tmp[:,2]
                grid_complete[:,0]=grid_tmp[:,0]
                grid_complete[:,1]=grid_tmp[:,1]
                grid_complete[:,2]=grid_tmp[:,2]
            else:
                grid_final[:,2]+=grid_tmp[:,2]
                grid_complete[:,2+i]=grid_tmp[:,2]
            i+=1
        grid_complete[:,-1]=np.sum(grid_complete[:,2:-2],axis=1)

        self.potential_grid=grid_final
        self.potential_grid_complete=grid_complete
        self.potential_grid_exist=True
        self.dynamic_components_last=copy.copy(self.dynamic_components)


        if show_comp==False:
            if Dgrid==False:
                grid_output=grid_final
            else:
                nrow,ncol=grid_final.shape
                grid_output=np.zeros(shape=(ncol,len(R),len(Z)))
                for i in range(ncol):
                    grid_output[i,:,:]=grid_final[:,i].reshape(len(R),len(Z))
        else:
            if Dgrid==False:
                grid_output=grid_complete
            else:
                nrow,ncol=grid_complete.shape
                grid_output = np.zeros(shape=(ncol, len(R), len(Z)))
                for i in range(ncol):
                    grid_output[i,:,:]=grid_complete[:,i].reshape(len(R),len(Z))


        return grid_output


    def save(self,filename,complete=True):


        if complete: save_arr=self.potential_grid_complete
        else: save_arr=self.potential_grid

        if self.potential_grid_exist:

            if complete:
                header=''
                header+='0-R 1-Z'

                i=2
                for comp in self.dynamic_components_last:
                    header+=' %i-%s'%(i,comp.name)
                    i+=1
                header+=' %i-External %i-Total'%(i,i+1)
                save_arr = self.potential_grid_complete

            else:

                header='0-R 1-Z 2-Total'
                save_arr = self.potential_grid

        else:

            raise AttributeError('Potential grid does not exist, make it with potential method')

        footer='R and Z in Kpc, Potentials in Kpc^2/Myr^2\n'
        i=0
        for comp in self.dynamic_components_last:
            footer += '*****************\n'
            footer += 'Component %i \n'%i
            footer += comp.__str__()
            i += 1
        footer += '*****************'

        np.savetxt(filename,save_arr,fmt='%.5e',header=header,footer=footer)


    def vcirc(self,R,nproc=2,toll=1e-4,show_comp=True):

        ncomp=len(self.dynamic_components)

        if show_comp:
            ret_array=np.zeros((len(R),ncomp+2))
        else:
            ret_array=np.zeros((len(R),2))

        ret_array[:,0]=R


        i=1
        v_tot2=0
        for comp in self.dynamic_components:
           v_tmp=comp.vcirc(R,nproc=nproc,toll=toll)[:,-1]
           v_tot2+=v_tmp*np.abs(v_tmp)


           if show_comp:
               ret_array[:,i]=v_tmp

           i+=1

        ret_array[:, -1] = np.sqrt(v_tot2)

        return ret_array


    def dens(self,R,Z=0,grid=False,show_comp=True, output='1D'):


        if output=='1D': Dgrid=False
        elif output=='2D': Dgrid=True
        else: raise NotImplementedError('output type \'%s\' not implemented for disc.potential'%str(output))

        ncomp=len(self.dynamic_components)

        comp0=self.dynamic_components[0]

        ret0=comp0.dens(R=R,Z=Z, grid=grid,output=output)

        if ncomp==1:

            ret_array=ret0

        else:


            if Dgrid:
                if show_comp:
                    ret_array=np.zeros(shape=(ncomp+3,len(R),len(Z)))
                    ret_array[2,:,:]=ret0[2,:,:]
                else:
                    ret_array=np.zeros(shape=(3,len(R),len(Z)))

                ret_array[0,:,:]=ret0[0,:,:]
                ret_array[1,:,:]=ret0[1,:,:]
                ret_array[-1,:,:]=ret0[2,:,:]

                i=3
                for comp in self.dynamic_components[1:]:
                    dens_tmp=comp.dens(R=R,Z=Z,grid=grid, output=output)[2,:,:]
                    if show_comp: ret_array[i,:,:]=dens_tmp
                    ret_array[-1,:,:]=ret_array[-1,:,:]+dens_tmp
                    i+=1


            else:
                if show_comp:
                    ret_array=np.zeros((len(ret0),ncomp+3))
                    ret_array[:, 2] = ret0[:, 2]
                else:
                    ret_array=np.zeros((len(ret0),3))

                ret_array[:,0]=ret0[:,0]
                ret_array[:,1]=ret0[:,1]
                ret_array[:,-1]=ret0[:,2]


                i=3
                for comp in self.dynamic_components[1:]:

                    dens_tmp=comp.dens(R=R,Z=Z,grid=grid, output=output)[:,-1]
                    if show_comp: ret_array[:,i]=dens_tmp
                    ret_array[:,-1]=ret_array[:,-1]+dens_tmp
                    i+=1

        return ret_array



    def dynamic_components_info(self):

        i=0
        print('Number of dynamical components: ',self.ncomp)
        for comp in self.dynamic_components:

            print('Components:',i)
            print(comp)
            i+=1