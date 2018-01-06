import xlrd
#import csv
import numpy as np

#FILENAME = ['/Users/josephmahoney/Documents/ML-tf/R5330 1 of 9.xlsx',
#'/Users/josephmahoney/Documents/ML-tf/R5330 2 of 9.xlsx']

def conv(fn): #pass a SINGLE file name
    xl = xlrd.open_workbook(fn)
    sh = xl.sheet_by_name('Results')
    well = []
    size_bp = [] #needs to be a list of lists
    conc_rel = [] #needs to be a list of lists
    size_loader = []
    conc_loader = []
    for rownum in range(sh.nrows):
        if sh.row_values(rownum)[0] != "": #makes list of well locations
            raw_word = str(sh.row_values(rownum)[0]).split()
            if raw_word[0] == "Capillary":
                well.append(raw_word[-1])
        if (type(sh.row_values(rownum)[1]) == float):
            size_loader.append(sh.row_values(rownum)[1])
            conc_loader.append(sh.row_values(rownum )[2])
            if (rownum+1) == sh.nrows: #loads if last cell in sheet "H12"
                size_bp.append(size_loader)
                conc_rel.append(conc_loader)
                size_loader = []
                conc_loader = []
        elif (sh.row_values(rownum)[1] == "Size (b.p.)") and (sh.row_values(rownum + 1)[1] == ""):
            #in case the capliary has no values a blank list is added to size_bp, conc_rel
            size_loader = []
            conc_loader = []
            size_bp.append(size_loader)
            conc_rel.append(conc_loader)
        elif (type(sh.row_values(rownum)[1]) != float) and (len(size_loader) > 0):
            size_bp.append(size_loader)
            conc_rel.append(conc_loader)
            size_loader = []
            conc_loader = []
    assert len(size_bp) == len(conc_rel)
    assert len(well) == len(size_bp)
    return well, size_bp, conc_rel

def make_array(fl): #pass a LIST of filenames
    zero_array = np.zeros(((len(fl)*96), 600))
    for f in range(len(fl)):
        wells, size, conc = conv(fl[f])
        for well in range(len(wells)): #index of each well
            for peak in range(len(size[well])): #index of each peak in a well. returns int
                conc_well = conc[well] #list of concentrations in each well.  returns list
                loc = size[well] #list of locations for each peak in a well.  returns list
                if (loc[peak] < 600) and (loc[peak] >= 0):
                    zero_array[well+(f*96), int(loc[peak])] = conc_well[peak] #works! zero at beginning of each array row.... for location zero
    return(zero_array) # this is actually an array full of zeros and a few peaks

#data = make_array(FILENAME)
#print(data.shape)
#print(data)
