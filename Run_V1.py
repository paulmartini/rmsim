import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_LCs as run
import txt_gen as txt
import jav_run as jav


                    #######################################
                    #        Start of Simulations         #
                    #######################################

                              ##################
                              # Read in inputs #
                              ##################
'''
Program takes in inputs in the form of the following:

    - a comma seperated txt file that contains 4 columns:
        - ID/name of source
        - g-Band magnitude
        - redshift
        - Standard deviation of photometric data
        
    - The number of realisations of each test, this will be the number of
        lightcurves per .fits file.
        
    - Array of lag fractions.
    
    - Files that contain as their first column, the observational cadence to be
        used for the subsampling for each source.
        
    - A file that contains the following columns:
        - ID/name of source
        - Mean photometric error
        - Mean Spectroscopic error
        - Photometric error standard devation
        - Spectroscopic error standard deviation
'''
# Physical characteristics of each source
source_phys_stats = pd.read_csv('Janies.txt',sep='\t')

# Number of realisations for each lightcurve
num_real = 100

# Array of lag fractions (fraction of R-L relation lag)
lag_fracs = [0.25,0.5,0.75,1.,1.25,1.5]

# Cadence file directory
cad_dir = 'Processed_LC/'

#curve stats
stats = pd.read_csv('Stats/stats_all.txt',sep='\t',names=['ID','photo_err','line_err','photo_err_std','line_err_std'])

txt_flag = False
fits_flag = False
Jav_flag = True

                      ####################################
                      #  Create simuulated light curves  #
                      ####################################

'''
Run simulations using mock_LC codes, output .fits file. Takes as input:
    - source_phys_stats (first text file discribed above)
    -

    
'''
if fits_flag == True:
    for i in range(len(source_phys_stats['NAME'])):
        
        cad_photo = cad_dir+ str(source_phys_stats['NAME'][i]) +'_gBand.txt'
        cad_spec = cad_dir+ str(source_phys_stats['NAME'][i]) +'_line.txt'
        
        run.create_fits(np.array(source_phys_stats.iloc[i,:]),lag_fracs, [cad_photo,cad_spec],num_real)



                     ######################################
                     #  Create text files for Jav + ICCF  #
                     ######################################

'''
Run function that creates .txt files from .fits, enforces cadence and adjusts
    errors. Takes as input:

'''
if txt_flag == True:

    for i in range(len(source_phys_stats['NAME'])):
        #Get error stats for each source
        try:
            source_stats = np.array(stats.loc[stats['ID'] == source_phys_stats['NAME'][i]])
        except:
            source_stats = [source_phys_stats['NAME'][i],0.03,0.15,0.02,0.075]
            
        print(source_stats)

        # generate txt files
        txt.txt_gen(source_stats[0],np.array(source_phys_stats.iloc[i,:]),lag_fracs, num_real)



                          ##########################
                          #  Run recovery methods  #
                          ##########################
'''
Run recovery methods with flag for Javelin. Takes as input: ###
    
'''
if Jav_flag == True:
    for i in range(len(source_phys_stats['NAME'])):
        jav.run_jav(source_phys_stats['NAME'][i],lag_fracs,num_real,150,500)


                             ###################
                             #  Do statistics  #
                             ###################

'''
This section will give outputs showing that your simulated curves are statistically consistent 
with your data. This will be in the form of a number of comparitive histograms for standard 
deviation, mean, variance etc.
'''

                            #####################
                            #  Produce Figures  #
                            #####################
'''
    
'''



