import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

from bhmtorch_cpu import BHM_VELOCITY_PYTORCH

def save_mdl(args, model, path, train_time):
    """
    @param model: BHM Module to save
    @param path (str): path relative to the mdl folder to save to
    """
    mdl_type = type(model).__name__
    print(" mdl_type:", mdl_type)

    print(" Saving as ./mdls/velocity/{}".format(path))
    if not os.path.isdir("./mdls/velocity/"):
        os.makedirs('./mdls/velocity/')

    if args.likelihood_type == "gamma":
        torch.save({
            'grid': model.grid,
            "w_hatx":model.w_hatx,
            "w_haty":model.w_haty,
            "w_hatz":model.w_hatz,
            "likelihood_type":model.likelihood_type,
            'train_time': train_time,
            }, "./mdls/velocity/{}".format(path)
        ) ###///###
    elif args.likelihood_type == "gaussian":
        torch.save({
            'mu_x': model.mu_x,
            'sig_x': model.sig_x,
            'mu_y': model.mu_y,
            'sig_y': model.sig_y,
            'mu_z': model.mu_z,
            'sig_z': model.sig_z,
            'grid': model.grid,
            'alpha': model.alpha,
            'beta': model.beta,
            'likelihood_type': model.likelihood_type,
            'train_time': train_time,
            }, "./mdls/velocity/{}".format(path)
        )
    else:
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

def train_velocity(args, alpha, beta, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    totalTime = 0
    # filter X,y such that only give the X's where y is 1

    if args.likelihood_type == "gamma":
        bhm_velocity_mdl = BHM_VELOCITY_PYTORCH(
            gamma=args.gamma,
            grid=None,
            cell_resolution=cell_resolution,
            cell_max_min=cell_max_min,
            X=X,
            nIter=1,
            kernel_type=args.kernel_type,
            likelihood_type=args.likelihood_type
        )
    elif args.likelihood_type == "gaussian":
        bhm_velocity_mdl = BHM_VELOCITY_PYTORCH(
            gamma=args.gamma,
            alpha=alpha,
            beta=beta,
            grid=None,
            cell_resolution=cell_resolution,
            cell_max_min=cell_max_min,
            X=X,
            nIter=1,
            kernel_type=args.kernel_type,
            likelihood_type=args.likelihood_type
        )
    else:
        raise ValueError(" Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    time1 = time.time()
    bhm_velocity_mdl.fit(X, y_vx, y_vy, y_vz, eps=0) # , y_vy, y_vz
    train_time = time.time() - time1
    print(' Total training time={} s'.format(round(train_time, 2)))
    save_mdl(args, bhm_velocity_mdl, '{}_f{}'.format(args.save_model_path, framei), train_time)
    del bhm_velocity_mdl
