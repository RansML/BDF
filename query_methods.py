import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

from bhmtorch_cpu import BHM_VELOCITY_PYTORCH
from utils_filereader import read_frame_velocity
from utils_metrics import calc_scores_velocity


def load_mdl(args, path):
    """
    @param path (str): path relative to the mdl folder to load from
    @returns: model: BHM Module that is loaded
    """
    filename = './mdls/{}'.format(path)
    print(' Loading the trained model from ' + filename)
    model_params = torch.load(filename)

    if args.likelihood_type == "gamma":
        model = BHM_VELOCITY_PYTORCH(
            gamma=args.gamma,
            grid=model_params['grid'],
            w_hatx=model_params["w_hatx"],
            w_haty=model_params["w_haty"],
            w_hatz=model_params["w_hatz"],
            kernel_type=args.kernel_type,
            likelihood_type=model_params["likelihood_type"]
        )
    elif args.likelihood_type == "gaussian":
        model = BHM_VELOCITY_PYTORCH(
            alpha=model_params['alpha'],
            beta=model_params['beta'],
            gamma=args.gamma,
            grid=model_params['grid'],
            kernel_type=args.kernel_type,
            likelihood_type=model_params["likelihood_type"]
        )
        model.updateMuSig(model_params['mu_x'], model_params['sig_x'],
                          model_params['mu_y'], model_params['sig_y'],
                          model_params['mu_z'], model_params['sig_z'])
    else:
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    return model, model_params['train_time']


def save_query_data(data, path):
    """
    @param data (tuple of elements): datapoints from regression/occupancy query to save
    @param path (str): path relative to query_data folder to save data
    """

    ###===###
    complete_dir = './query_data/{}'.format(path).split("/")
    complete_dir = "/".join(complete_dir[:-1])

    if not os.path.isdir(complete_dir):
        os.makedirs(complete_dir)
    ###///###

    filename = './query_data/{}'.format(path)
    torch.save(data, filename)
    print( ' Saving queried output as ' + filename)

def query_velocity(args, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    bhm_velocity_mdl, train_time = load_mdl(args, 'velocity/{}_f{}'.format(args.save_model_path, framei))

    option = ''
    if args.eval_path != '' and args.eval:
        #if eval is True, test the query
        print(" Query data from the test dataset")
        Xq_mv, y_vx_true, y_vy_true, y_vz_true, _ = read_frame_velocity(args, framei, args.eval_path, cell_max_min)
        option = args.eval_path
    elif args.query_dist[0] <= 0 and args.query_dist[1] <= 0 and args.query_dist[2] <= 0:
        #if all q_res are non-positive, then query input = X
        print(" Query data is the same as input data")
        Xq_mv = X
        option = 'Train data'
    elif args.query_dist[0] <= 0 or args.query_dist[1] <= 0 or args.query_dist[2] <= 0:
        #if at least one q_res is non-positive, then
        if args.query_dist[0] <= 0: #x-slice
            print(" Query data is x={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'X slice at '.format(args.query_dist[3])
        elif args.query_dist[1] <= 0: #y-slice
            print(" Query data is y={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Y slice at '.format(args.query_dist[3])
        else: #z-slice
            print(" Query data is z={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Z slice at '.format(args.query_dist[3])
    else:
        #if not use the grid
        print(" Query data is a 3D grid.")
        xx, yy, zz = torch.meshgrid(
            torch.arange(
                cell_max_min[0],
                cell_max_min[1]+args.query_dist[0],
                args.query_dist[0]
            ),
            torch.arange(
                cell_max_min[2],
                cell_max_min[3]+args.query_dist[1],
                args.query_dist[1]
            ),
            torch.arange(
                cell_max_min[4],
                cell_max_min[5]+args.query_dist[2],
                args.query_dist[2]
            )
        )
        Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        option = '3D grid'

    time1 = time.time()

    if args.likelihood_type == "gamma":
        mean_x, mean_y, mean_z = bhm_velocity_mdl.predict(Xq_mv)
    elif args.likelihood_type == "gaussian":
        mean_x, var_x, mean_y, var_y, mean_z, var_z = bhm_velocity_mdl.predict(Xq_mv, args.query_blocks, args.variance_only)
    else:
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    query_time = time.time() - time1

    print(' Total querying time={} s'.format(round(query_time, 2)))
    save_query_data((X, y_vx, y_vy, y_vz, Xq_mv, mean_x, var_x, mean_y, var_y, mean_z, var_z, framei), \
                    'velocity/{}_f{}'.format(args.save_query_data_path, framei))

    if args.eval:
        if hasattr(args, 'report_notes'):
            notes = args.report_notes
        else:
            notes = ''
        axes = [('x', y_vx_true, mean_x, var_x), ('y', y_vy_true, mean_y, var_y), ('z', y_vz_true, mean_z, var_z)]
        for axis, Xqi, mean, var in axes:
            mdl_name = 'reports/' + args.plot_title + '_' + axis
            calc_scores_velocity(mdl_name, option, Xqi.numpy(), mean.numpy().ravel(), predicted_var=\
                np.diagonal(var.numpy()), train_time=train_time, query_time=query_time, save_report=True, notes=notes)
