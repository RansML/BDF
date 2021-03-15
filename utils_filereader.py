import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

# ==============================================================================
# Utility Functions
# ==============================================================================
def format_config(args):
    """
    Formats default parameters from argparse to be easily digested by module
    """
    # default parameter to reduce min and max bounds of cell_max_min
    delta = (args.area_max[0] - args.area_min[0])*0.03

    fn_train = os.path.abspath(args.dataset_path)
    cell_resolution = (
        args.hinge_dist[0],
        args.hinge_dist[1],
        args.hinge_dist[2]
    )
    cell_max_min = [
        args.area_min[0] + delta,
        args.area_max[0] - delta,
        args.area_min[1] + delta,
        args.area_max[1] - delta,
        args.area_min[2] + delta,
        args.area_max[2] - delta
    ]
    return fn_train, cell_max_min, cell_resolution

def get3DPartitions(args, cell_max_min, nPartx1, nPartx2, nPartx3):
    """
    @param cell_max_min: The size of the entire area
    @param nPartx1: How many partitions along the longitude
    @param nPartx2: How many partitions along the latitude
    @param nPartx3: How many partition along the altitude
    @return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    length = cell_max_min[3] - cell_max_min[2]
    height = cell_max_min[5] - cell_max_min[4]

    x_margin = width/2
    y_margin = length/2
    z_margin = height/2

    x_partition_size = width/nPartx1
    y_partition_size = length/nPartx2
    z_partition_size = height/nPartx3
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            for z in range(nPartx3):
                seg_i = (
                    cell_max_min[0] + x_partition_size*(x-args.partition_bleed),  # Lower X
                    cell_max_min[0] + x_partition_size*(x+1+args.partition_bleed),  # Upper X
                    cell_max_min[2] + y_partition_size*(y-args.partition_bleed),  # Lower Y
                    cell_max_min[2] + y_partition_size*(y+1+args.partition_bleed),  # Upper Y
                    cell_max_min[4] + z_partition_size*(z-args.partition_bleed),  # Lower Z
                    cell_max_min[4] + z_partition_size*(z+1+args.partition_bleed)  # Upper Z
                )
                cell_max_min_segs.append(seg_i)
    return cell_max_min_segs

def read_frame(args, framei, fn_train, cell_max_min):
    """
    @params: framei (int) - the index of the current frame being read
    @params: fn_train (str) - the path of the dataset frames being read
    @params: cell_max_min (tuple of 6 ints) - bounding area observed

    @returns: g (float32 tensor)
    @returns: X (float32 tensor)
    @returns: y_occupancy (float32 tensor)
    @returns: sigma (float32 tensor)
    @returns: cell_max_min_segments - partitioned frame data for frame i

    Reads a single frame (framei) of the dataset defined by (fn_train) and
    and returns LIDAR hit data corresponding to that frame and its partitions
    """
    print(' Reading '+fn_train+'.csv...')
    g = pd.read_csv(fn_train+'.csv', delimiter=',')
    g = g.loc[g['t'] == framei].values[:, 2:]

    g = torch.tensor(g, dtype=torch.float32)
    X = g[:, :3]
    y_occupancy = g[:, 3].reshape(-1, 1)
    sigma = g[:, 4:]

    # If there are no defaults, automatically set bounding area.
    if cell_max_min[0] == None:
        cell_max_min[0] = X[:,0].min()
    if cell_max_min[1] == None:
        cell_max_min[1] = X[:,0].max()
    if cell_max_min[2] == None:
        cell_max_min[2] = X[:,1].min()
    if cell_max_min[3] == None:
        cell_max_min[3] = X[:,1].max()
    if cell_max_min[4] == None:
        cell_max_min[4] = X[:,2].min()
    if cell_max_min[5] == None:
        cell_max_min[5] = X[:,2].max()

    cell_max_min_segments = get3DPartitions(args, tuple(cell_max_min), args.num_partitions[0], args.num_partitions[1], args.num_partitions[2])
    return g, X, y_occupancy, sigma, cell_max_min_segments

def read_frame_velocity(args, framei, fn, cell_max_min):
    print(' Reading '+fn+'.csv...')
    g = pd.read_csv(fn+'.csv', delimiter=',')
    g = g.loc[np.isclose(g['t'],framei)].values[:, 1:]

    # g = torch.tensor(g, dtype=torch.double)
    g = torch.tensor(g, dtype=torch.float32)
    X = g[:, :3]
    y_vx = g[:, 3].reshape(-1, 1)
    y_vy = g[:, 4].reshape(-1, 1)
    y_vz = g[:, 5].reshape(-1, 1)

    # If there are no defaults, automatically set bounding area.
    if cell_max_min[0] == None:
        cell_max_min[0] = X[:,0].min()
    if cell_max_min[1] == None:
        cell_max_min[1] = X[:,0].max()
    if cell_max_min[2] == None:
        cell_max_min[2] = X[:,1].min()
    if cell_max_min[3] == None:
        cell_max_min[3] = X[:,1].max()
    if cell_max_min[4] == None:
        cell_max_min[4] = X[:,2].min()
    if cell_max_min[5] == None:
        cell_max_min[5] = X[:,2].max()

    cell_max_min_segments = get3DPartitions(args, tuple(cell_max_min), args.num_partitions[0], args.num_partitions[1], args.num_partitions[2])
    return X, y_vx, y_vy, y_vz, cell_max_min_segments
