import os
import numpy as np
import pandas as pd
import pptk

dataset_path = os.getcwd() + '/datasets/ahmed/ransvid1_600.csv'
fn_train = os.path.abspath(dataset_path)
data = pd.read_csv(fn_train).to_numpy() #['', 't', 'X', 'Y', 'Z', 'occupancy', 'sig_x', 'sig_y', 'sig_z']
print(data)

v = pptk.viewer(data[:,2:5])
v.attributes(data[:,5],)
v.set(point_size=1)

