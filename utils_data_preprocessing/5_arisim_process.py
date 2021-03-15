import pandas as pd
import numpy as np
import pptk
import sys

# Settings
mode = 'pos' #vel, pos, chris
fn_in_dir = '/home/ransalu/Desktop/PycharmProjects/SpaTUn/datasets/5_airsim/5_airsim1/'
fn_out_dir = '/home/ransalu/Desktop/PycharmProjects/SpaTUn/datasets/5_airsim/5_airsim1/'

df0 = pd.read_csv(fn_in_dir + 'formatted_data0.csv').to_numpy()
df1 = pd.read_csv(fn_in_dir + 'formatted_data1.csv').to_numpy()
df2 = pd.read_csv(fn_in_dir + 'formatted_data2.csv').to_numpy()
df3 = pd.read_csv(fn_in_dir + 'formatted_data3.csv').to_numpy()


traj = np.concatenate((0*np.ones(df0.shape[0]), 1*np.ones(df1.shape[0]), 2*np.ones(df2.shape[0]), 3*np.ones(df3.shape[0])))
txyzv3i = np.vstack((df0, df1, df2, df3))
txyzv3i = np.hstack((txyzv3i, traj[:, None]))

txyzv3i[:, 0] = 0.0
txyzv3i[:, 3] *= -6
txyzv3i[:, 1:4] /= 100

if mode == 'vel':
    np.savetxt(fn_out_dir + '5_airsim1_vel.csv', txyzv3i, delimiter=',', header='t, X, Y, Z, V_X, V_Y, V_Z, traj_id', comments='')#, fmt=' '.join(['%i'] + ['%1.8f']*7))
elif mode == 'pos':
    txyzv3i[:, 4:7] = txyzv3i[:, 1:4]
    np.savetxt(fn_out_dir + '5_airsim1_pos.csv', txyzv3i, delimiter=',', header='t, X, Y, Z, X, Y, Z, traj_id', comments='')#, fmt=' '.join(['%i'] + ['%1.8f']*7))


print(mode, 'array size', txyzv3i.shape)
print("min: {}".format(txyzv3i.min(axis=0)))
print("max: {}".format(txyzv3i.max(axis=0)))
print("mean: {}".format(txyzv3i.mean(axis=0)))

v0 = pptk.viewer(txyzv3i[:, 1:4], txyzv3i[:, 4], txyzv3i[:, 5], txyzv3i[:, 6], txyzv3i[:, 7])
v0.color_map('jet')#, scale=[0,20])
#v0.set(point_size=0.01)