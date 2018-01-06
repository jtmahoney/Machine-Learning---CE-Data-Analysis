import numpy as np


FILES = ['']

def read_to_array(file):
    pf = []
    for f in file:
        tabl = np.genfromtxt(f, delimiter=',', usecols=3, dtype=str, skip_header=1)
        pf.extend(tabl) #adds tabl list to pf EXTEND!!!
    arra = np.array([pf])
    arra = np.transpose(arra)
    array_coded = np.where(arra == 'Pass', 1, 0)
    return array_coded

#print(read_to_array(FILES))
