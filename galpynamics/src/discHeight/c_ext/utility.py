import numpy as np


#####Calculate dens
def cdens(table,disp):
    """
    Calculate the  normalized vertical density given a table with R-Z-Pot.
    The d/d0 is calculated from the hydrostatical equlibrium as Exp(-(pot(R,Z)-pot(R,Zmin)/disp^2(R))
    NB. the table need to be sorted both in R and Z
    NB2. the reference value at pot(R,Zmin) is the value of the potential for the minimum Z found in the file
    :param table: 3 column numpy array
    :param disp: disp function of radius  (km/s)
    :return: a new numpy array with 4 column, the first three are the same of the input table, the last
            represents the normalized density d/d0 at coordinates R,Z.
    """

    conv= 1.02269012107e-3 #km/s to kpc/Myr
    newtable=np.zeros(shape=(len(table),4),dtype=np.float64,order='C')
    newtable[:,0:3]=table
    count=0
    i=0
    new=0
    while new!=table[-1,0]:
        old=table[i,0]
        new=table[i+1,0]
        if (new==old): count+=1
        else:
            r=newtable[i-count,0]
            pot0=newtable[i-count,2]
            newtable[i-count:i+1,3]=np.exp(-(newtable[i-count:i+1,2]-pot0)/(disp(r)*disp(r)*conv*conv))
            count=0
        i+=1
    r=newtable[i,0]
    pot0=newtable[i,2]
    newtable[i:,3]=np.exp(-(newtable[i:,2]-pot0)/(disp(r)*disp(r)*conv*conv))
    newtable[:,3]=np.where(newtable[:,3]/newtable[0,3]>1,np.nan,newtable[:,3])

    return newtable