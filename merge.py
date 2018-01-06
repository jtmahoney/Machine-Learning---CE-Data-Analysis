# Joseph Mahoney 23Aug2017
# merge.py loads data from 600mer CE and meges them in order to feed data into
# machine learning classifyer for expediting 600mer CE data analysis.

import pandas as pd

wells = {'A1':1, 'A2':2}
well_data = {}
#error .csv file exremely easy to read
err_df = pd.read_csv('plate_1_errors.csv')
print "error .csv file loaded"
#had to strip headers becase of the layout of this file... total garbage... thanks AATI
peak_df = pd.read_csv('plate_1_peak_table.csv', header=None)
print "peak .csv file loaded"

for well in wells:
    well_loader = []
    well_string = ('Capillary ' + str(wells[well]) + ': ' + well + ': ' + well)
    start = peak_df.loc[peak_df[0]==well_string].index[0]
    peak_size = {}
    peak_conc = {}
    iterator = 2
    #print start
    #print (peak_df.iat[(start + 2) ,0])
    #print (peak_df.iat[(start + iterator + 2), 0])
    #print (type(peak_df.iat[(start + iterator), 0]))

    while True: #iterates starting at start + iterator and continues untill is find a string 'nan'
                #this could backfire badly.... should handle if 'nan' isnt the default null in csv file
        if str(peak_df.iat[(start + iterator), 0]) != 'nan':
            peak_size[(peak_df.iat[(start + iterator), 0])] = peak_df.iat[(start + iterator), 1]
            peak_conc[(peak_df.iat[(start + iterator), 0])] = peak_df.iat[(start + iterator), 2]
            iterator = iterator + 1
        else:
            break

    well_loader.append(peak_size)
    well_loader.append(peak_conc)
    well_data[well] = well_loader
    #print peak_size
print well_data
