import numpy as np
import pickle
import pymap3d as pm
import matplotlib.pyplot as pl
import pptk
import sys

# Settings
mode = 'vel' #vel, pos, chris
fn_in_dir = '/home/ransalu/Desktop/PycharmProjects/JFK_dataset/'
fn_out_dir = '/home/ransalu/Desktop/PycharmProjects/SpaTUn/datasets/6_jfk/'

# Load trajectories
# The files contains extracted trajectories flying within 30NM from JFK airport.
# * jfk_trajs_set_1.pkl : 20120501 ~ 20120505
# * jfk_trajs_set_2.pkl : 

traj_data = pickle.load(open(fn_in_dir+'jfk_trajs_set_1.pkl', 'rb'))
traj_data = np.array(traj_data)
print('total number of trajectories=', len(traj_data))
print('first traj shape=', traj_data[0].shape) # shape of first(sample) trajectory

# Draw 2D plot
M_TO_NM = 0.000539957
M_TO_FT = 3.28084
FT_TO_M = .3048

airport_lat, airport_lon, airport_altitude = 40.63993, -73.77869, 12.7
jfk_runways = [[[40.62202, -73.78558], [40.65051, -73.76332]],
               [[40.62543, -73.77035], [40.64524, -73.75486]],
               [[40.65776, -73.79025], [40.64372, -73.75928]],
               [[40.64836, -73.81671],[40.62799, -73.77178]]] #04L-22R, 04R-22L, 13L-31R, 13R-31L 

def plot_rwy(rwy_coords, color):
    for r in rwy_coords:
        a,b = r[0],r[1]
        start_x, start_y, _ = pm.geodetic2enu(a[0], a[1], airport_altitude, airport_lat, airport_lon, airport_altitude)
        end_x, end_y, _ = pm.geodetic2enu(b[0], b[1], airport_altitude, airport_lat, airport_lon, airport_altitude)
        pl.plot([start_x*M_TO_NM, end_x*M_TO_NM], [start_y*M_TO_NM, end_y*M_TO_NM], '-', c=color, lw=2)


traj_data_subset = traj_data[:100]
for i, traj in enumerate(traj_data_subset):
    t = traj[:, 0]  #time-step
    diff = t[1:] - t[:-1]
    pos = traj[:, 1:4]  #position [xEast(m), yNorth(m), zUp(m)]
    pos[:,:2] /= 25 #TODO scaled
    v = np.sqrt(np.sum((pos[1:, :] - pos[:-1,:])**2, axis=1)) / diff
    vx = (pos[1:, 0] - pos[:-1, 0]) / diff
    vy = (pos[1:, 1] - pos[:-1, 1]) / diff
    vz = (pos[1:, 2] - pos[:-1, 2]) / diff
    #print(v.min(), v.max(), v.mean())

    if mode == 'vel':
        temp = np.hstack((t[1:,None], pos[1:,:], vx[:,None], vy[:,None], vz[:,None], i+0*vx[:,None])) #velocity
    elif mode == 'pos':
        temp = np.hstack((t[1:, None], pos[1:, :], pos[1:, :], i + 0 * vx[:, None])) #position
    if i == 0:
        txyzv3i = temp
    else:
        txyzv3i = np.vstack((txyzv3i, temp))

# filtering area
delta = 500
cond1 = np.logical_and(txyzv3i[:,1]<=delta, txyzv3i[:,1]>=-delta)
cond = np.logical_and(cond1, np.logical_and(txyzv3i[:,2]<=delta, txyzv3i[:,2]>=-delta))
txyzv3i = txyzv3i[cond,:]
txyzv3i[:,1:4] /= 250 #TODO scaled to be in +-2
txyzv3i[:,0] = 0

pl.figure(figsize=(14,3))
pl.subplot(131)
pl.scatter(txyzv3i[:,1], txyzv3i[:,2], c=txyzv3i[:,3], s=2, cmap='jet'); pl.colorbar(); pl.title('xy')
pl.subplot(132)
pl.scatter(txyzv3i[:,1], txyzv3i[:,3], c=txyzv3i[:,2], s=2, cmap='jet'); pl.colorbar(); pl.title('xz')
pl.subplot(133)
pl.scatter(txyzv3i[:,2], txyzv3i[:,3], c=txyzv3i[:,1], s=2, cmap='jet'); pl.colorbar(); pl.title('yz')
pl.show()
#sys.exit()

np.set_printoptions(precision=2)
print(mode, 'array size', txyzv3i.shape)
print("min: {}".format(txyzv3i.min(axis=0)))
print("max: {}".format(txyzv3i.max(axis=0)))
print("mean: {}".format(txyzv3i.mean(axis=0)))
v = pptk.viewer(txyzv3i[:,1:4], txyzv3i[:,4], txyzv3i[:,5], txyzv3i[:,6], txyzv3i[:,7])
v.color_map('jet')#, scale=[0,20])

if mode == 'chris':
    noise = np.random.normal(0, 2, size=txyzv3i.shape)
    txyzv3i += noise
    np.save(fn_out_dir + 'chris_3d_dataset.npy', txyzv3i)
elif mode == 'pos':
    fn_out = fn_out_dir + '6_jfk_partial1_pos.csv'
    np.savetxt(fn_out, txyzv3i, delimiter=',', header='t, X, Y, Z, X, Y, Z, traj_id',
               comments='')  # , fmt=' '.join(['%i'] + ['%1.8f']*7))
elif mode == 'vel':
    fn_out = fn_out_dir + '6_jfk_partial1_vel.csv'
    np.savetxt(fn_out, txyzv3i, delimiter=',', header='t, X, Y, Z, V_X, V_Y, V_Z, traj_id',
               comments='')  # , fmt=' '.join(['%i'] + ['%1.8f']*7))

sys.exit()


# plot whole set of trajectories
pl.figure(figsize=(7,7))
pl.xlabel('East (NM)'); pl.ylabel('North (NM)')
pl.xlim([-30, 30]); pl.ylim([-30, 30])

for i, traj in enumerate(traj_data):
    t = traj[:, 0]  #time-step
    pos = traj[:, 1:4]  #position [xEast(m), yNorth(m), zUp(m)]
    
    pl.plot(pos[:,0]*M_TO_NM, pos[:,1]*M_TO_NM,'--b', lw=0.1)  #2D plot
plot_rwy(jfk_runways, color='r')
pl.show()


# plot subset of trajectories
traj_data_subset = traj_data[:10]

pl.figure(figsize=(7,7))
pl.xlabel('East (NM)'); pl.ylabel('North (NM)')
pl.xlim([-30, 30]); pl.ylim([-30, 30])

for i, traj in enumerate(traj_data_subset):
    t = traj[:, 0]  #time-step
    pos = traj[:, 1:4]  #position [xEast(m), yNorth(m), zUp(m)]
    
    pl.plot(pos[:,0]*M_TO_NM, pos[:,1]*M_TO_NM,'--b', lw=0.5)  #2D plot
plot_rwy(jfk_runways, color='r')
pl.show()





