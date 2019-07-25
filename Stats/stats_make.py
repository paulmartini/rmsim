import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def variability(flux, error):
    # this is defined by Fausnaugh 2016
    
    variance = np.square(error)
    
    mean_flux = np.nanmean(flux)
    mean_variance = np.nanmean(variance)
    number = len(flux)
    
    fluxSum = 0
    for n in range(number):
        fluxSum = fluxSum + pow(flux[n] - mean_flux, 2) - variance[n]
    #print(fluxSum)
    flux_var = (1/mean_flux)*pow(abs(fluxSum)/number, 0.5)

    var_var = pow(mean_variance/(pow(mean_flux,2)*flux_var*pow(2*number, 0.5)),2) + \
    pow(pow(mean_variance/number, 0.5)/mean_flux, 2)
    
    sigma_var = pow(var_var, 0.5)
    
    snr_var = flux_var/sigma_var
    
    return flux_var, sigma_var, snr_var

CIV = pd.read_csv('CIV_noiseCut.dat',sep=' ')
MgII = pd.read_csv('MgII_noiseCut.dat',sep=' ')
Hbeta = pd.read_csv('Hbeta_noiseCut.dat',sep=' ')

print()

out = open('comp_stats.txt','w')

out.write('Source_ID\tmean_photo\tstd_photo\tfvar_photo\tfvar_line'+ '\tmean_err_photo\tmean_err_line\tstd_err_photo\tstd_err_line\t'+ '\n')

for i in range(len(CIV['Source_ID'])):

    photo_source = '../Processed_LC/'+str(CIV['Source_ID'][i])+'_gBand.txt'
    photo = pd.read_csv(photo_source,sep='   ',names=['mjd','flux','err'])
    
    line_source = '../Processed_LC/'+str(CIV['Source_ID'][i])+'_line.txt'
    line = pd.read_csv(line_source,sep='   ',names=['mjd','flux','err'])
    
    flux_var, sigma_var, snr_var = variability(photo['flux'], photo['err'])
    line_flux_var, sigma_var, snr_var = variability(line['flux'], line['err'])
    
    mean_photo = float(np.mean(photo['flux']))
    std_photo = float(np.std(photo['flux']))
    mean_err_photo = float(np.mean(photo['err']/photo['flux']))
    std_err_photo = float(np.std(photo['err']/photo['flux']))

    out.write(str(CIV['Source_ID'][i])+'\t'+str(mean_photo)+'\t'+str(std_photo)+'\t'+str(flux_var)+ '\t'+ str(line_flux_var)+'\t'+str(mean_err_photo)+'\t'+ str(CIV['PerErr'][i]/100.)+'\t'+str(std_err_photo)+'\t'+str(CIV['ErrStd'][i]/100.)+'\n')

for i in range(len(MgII['Source_ID'])):
    photo_source = '../Processed_LC/'+str(MgII['Source_ID'][i])+'_gBand.txt'
    photo = pd.read_csv(photo_source,sep='   ',names=['mjd','flux','err'])
    
    flux_var, sigma_var, snr_var = variability(photo['flux'], photo['err'])
    line_flux_var, sigma_var, snr_var = variability(line['flux'], line['err'])
    
    mean_photo = float(np.mean(photo['flux']))
    std_photo = float(np.std(photo['flux']))
    mean_err_photo = float(np.mean(photo['err']/photo['flux']))
    std_err_photo = float(np.std(photo['err']/photo['flux']))
    
    out.write(str(MgII['Source_ID'][i])+'\t'+str(mean_photo)+'\t'+str(std_photo)+'\t'+str(flux_var)+ '\t'+ str(line_flux_var)+'\t'+str(mean_err_photo)+'\t'+str(MgII['PerErr'][i]/100.)+'\t'+str(std_err_photo)+'\t'+str(MgII['ErrStd'][i]/100.)+'\n')


for i in range(len(Hbeta['Source_ID'])):
    photo_source = '../Processed_LC/'+str(Hbeta['Source_ID'][i])+'_gBand.txt'
    photo = pd.read_csv(photo_source,sep='   ',names=['mjd','flux','err'])
    
    flux_var, sigma_var, snr_var = variability(photo['flux'], photo['err'])
    line_flux_var, sigma_var, snr_var = variability(line['flux'], line['err'])
    
    mean_photo = float(np.mean(photo['flux']))
    std_photo = float(np.std(photo['flux']))
    mean_err_photo = float(np.mean(photo['err']/photo['flux']))
    std_err_photo = float(np.std(photo['err']/photo['flux']))
    
    out.write(str(Hbeta['Source_ID'][i])+'\t'+str(mean_photo)+'\t'+str(std_photo)+'\t'+str(flux_var)+ '\t'+ str(line_flux_var)+'\t'+str(mean_err_photo)+'\t'+str(Hbeta['PerErr'][i]/100.)+'\t'+str(std_err_photo)+'\t'+str(Hbeta['ErrStd'][i]/100.)+'\n')
