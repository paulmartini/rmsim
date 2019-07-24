import numpy as np
from mockLC import create
import pandas as pd
import os


def create_fits(phys_stats, lag_fracs, cadence_files, nLC):

    # PathDef!
    # Location and name of files containing SDSS composite QSO template,
    # and DES filter response curves for r-band and i-band filters
    QSOtemp = 'Backend_data/sdssqsocomposite.txt'
    rFilter = 'Backend_data/DES_asahi_r.dat'
    iFilter = 'Backend_data/DES_asahi_i.dat'
    gFilter = 'Backend_data/DES_asahi_g.dat'

    
    
    name = int(phys_stats[0])
    gmag = phys_stats[1]
    z = phys_stats[2]
    sigma = phys_stats[3]

    cadence_photo = cadence_files[0]
    cadence_spec = cadence_files[1]
    

    dir_source = 'Data/'+ str(name)+'/'
    if not os.path.exists(dir_source):
        os.makedirs(dir_source)
    

    
    LClen = 5000

    for j in range(len(lag_fracs)):
        
        dir_fnl = dir_source+str(lag_fracs[j])+'/'
        if not os.path.exists(dir_fnl):
            os.makedirs(dir_fnl)
        # Initialise class
        AGN = create(QSOtemp, rFilter, gFilter, iFilter)
    
        savefile = dir_fnl+str(name)+'_'+str(lag_fracs[j])+'_sim.fits'
       
        # Create AGN and light-curve properties (0th extension)
        AGN.createLCproperties(z, gmag, sigma, str(name)+'_'+str(lag_fracs[j]), lag_fracs[j],dir_fnl)
        
        # Create underlying light-curves (1st extension)
        if z <= 0.62: # H beta
            AGN.createUnderlyingLC(savefile, LClen, nLC=nLC, TF='TH', emline='CIV',ID=name)
        elif z <= 1.8 and z[i]>0.62: # MgII
            AGN.createUnderlyingLC(savefile, LClen, nLC=nLC, TF='TH', emline='CIV',ID=name)
        else: # CIV
            AGN.createUnderlyingLC(savefile, LClen, nLC=nLC, TF='TH', emline='CIV',ID=name)

 
        sfile = cadence_spec #LCdir + str(AGNname[ind]) + '_CIV.txt'
        pfile = cadence_photo #LCdir + str(AGNname[ind]) + '_gBand.txt'
        pfile1, sfile1 = pfile, sfile

        AGN.LCwDatesFromObservations(savefile, pfile, sfile)

