import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits
import random as rn
import pandas as pd
import sys,os


class MockLightCurve(object):
    
    def __init__(self,filepath=None):
        assert (filepath is not None), "No file name is specified."
        self.filepath = filepath
        try:
            self.data = fits.open(self.filepath)
        except IOError:
            print("Error: file {0} could not be found".format(filepath))
            exit()
        
        data = fits.open(self.filepath)
        self.z = data[0].header['Z']
        self.L = data[0].header['L']
        self.lagCIV = data[0].header['AGNLAGC']
        self.lagMgII = data[0].header['AGNLAGM']
        self.lagHbeta = data[0].header['AGNLAGH']
        self.r_mag = data[0].header['RAPPMAG']
        self.mBH = data[0].header['MBH']
        
        self.underlyinglcs = data[1].data
        self.ucDim = np.shape(self.underlyinglcs)
        
        self.cadence = data[2].data
        self.cad1Dim = np.shape(self.cadence)



def txt_gen(stats, phys_stats, lag_fracs, num_reals):
    
    name = int(phys_stats[0])
    
    mean_err_cont = stats[1]
    std_err_cont = stats[3]
    mean_err_line = stats[2]
    std_err_line = stats[4]
    
    
    
    for i in range(len(lag_fracs)):
                   
        dir_fits ='Data/'+ str(name)+'/'+str(lag_fracs[i])+'/'
                   
        dir_txt = dir_fits+'txt/'
        if not os.path.exists(dir_txt):
            os.makedirs(dir_txt)


        mock = MockLightCurve(dir_fits+str(name)+'_'+str(lag_fracs[i])+'_sim.fits')
               
        for k in range(num_reals):

        #read in continuum flux from one
            mjd1 = np.arange(0,5000) #mock1.measuredCont[:, 0, 0]
            flux1 = mock.underlyinglcs[ 0,:, k]

            
            #read in line flux from the other
            mjd2 = np.arange(0,5000) #mock1.measuredCont[:, 0, 0]
            flux2 = mock.underlyinglcs[1,:,k]
            
            
            cadence_cont = mock.cadence[0,:]
            cadence_line = mock.cadence[1,:]
            
            cadence_cont[4000:5000] = 0
            
            out_cont = open(dir_txt+str(name)+'_'+str(k)+'_'+str(lag_fracs[i])+'_cont.txt','w')
            out_line = open(dir_txt+str(name)+'_'+str(k)+'_'+str(lag_fracs[i])+'_line.txt','w')
            
            for n in range(len(cadence_cont)):
                
                line_err = np.random.normal(mean_err_line,std_err_line)
                cont_err = np.random.normal(mean_err_cont,std_err_cont)
                
                line_shift = np.mean(flux2)*mean_err_line*np.random.normal(0,0.3)
                cont_shift = np.mean(flux1)*mean_err_cont*np.random.normal(0,0.3)
                
                if cadence_cont[n] == 1:
                    out_cont.write(str(mjd1[n])+ '\t' + str(flux1[n]+cont_shift) +'\t' + str(flux1[n]*(cont_err))+'\n')
                if cadence_line[n] ==1:
                    out_line.write(str(mjd2[n])+ '\t' + str((flux2[n]+line_shift)) +'\t' + str(flux2[n]*(line_err))+'\n')

            out_cont.close()
            out_line.close()
