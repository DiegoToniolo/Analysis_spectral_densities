import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import sys
from jackknife import Jackknife

class main:
    def __init__(self, pd:str, pj:str):
        df = open(pd, "r")
        jf = open(pj, "r")

        d = df.readlines()

        data = []
        for line in d:
            jack = np.zeros(25)
            for i in range(25):
                jack[i]  = float(jf.readline())
            data.append(Jackknife())
            data[-1].mean = float((line.split())[0])
            data[-1].jack = jack

        for t in data:
            print(t.mean, np.sqrt(t.variance()))
        
        df.close()
        jf.close()

if len(sys.argv) < 2:
    print("Usage " + sys.argv[0] + " mean_file jack_file")
    exit(0)

main(sys.argv[1], sys.argv[2])