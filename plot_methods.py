import argparse
import json
import os
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
import torch

#vivian
import plotly.figure_factory as ff
import trimesh
import time
import torch
import numpy as np
from skimage import measure
import utils_filereader

from plotly.subplots import make_subplots

from sklearn.metrics.pairwise import euclidean_distances

plotly.io.orca.config.executable = "/home/khatch/anaconda3/envs/hilbert/bin/orca"


# ==============================================================================
# BHM Plotting Class
# ==============================================================================
class BHM_PLOTTER():
    def __init__(self, args, plot_title, surface_threshold, variance_threshold, query_dist, occupancy_plot_type='scatter', plot_volumetric=False, plot_axis="x"):
        self.args = args
        self.plot_title = plot_title
        self.variance_threshold = variance_threshold
        self.surface_threshold = surface_threshold
        self.query_dist = query_dist
        self.occupancy_plot_type = occupancy_plot_type

        self.plot_volumetric = plot_volumetric
        self.plot_axis = plot_axis

        print(' Successfully initialized plotly plotting class')

    def _filter_predictions_velocity(self, X, y, var):
        """
        :param X: Nx3 position
        :param y: N values
        :return: thresholded X, y vals
        """

        # Filter -1 to 1
        min_filterout = X.max(dim=-1).values >= 1
        max_filterout = X.min(dim=-1).values <= -1
        mask = torch.logical_not(torch.logical_or(min_filterout, max_filterout))
        X = X[mask, :]
        y = y[mask, :]
        var = var[mask, :]

        if len(self.surface_threshold) == 1:
            mask = y.squeeze() >= self.surface_threshold[0]
        else:
            min_mask = y.squeeze() >= self.surface_threshold[0]
            max_mask = y.squeeze() <= self.surface_threshold[1]
            mask = torch.logical_and(min_mask, max_mask)

        X = X[mask, :]
        y = y[mask, :]
        var = var[mask, :]

        var_mask = var.squeeze(-1) <= self.variance_threshold
        X = X[var_mask, :]
        y = y[var_mask, :]
        var = var[var_mask, :]

        return X, y, var

    def _filter_predictions_velocity_where(self, X, y, var):
        """
        :param X: Nx3 position
        :param y: N values
        :return: thresholded X, y vals
        """

        # Filter -1 to 1
        min_filterout = X.max(dim=-1).values >= 1
        max_filterout = X.min(dim=-1).values <= -1
        mask = torch.logical_not(torch.logical_or(min_filterout, max_filterout))
        # X = torch.where((torch.ones_like(X) * mask[:, None]).to(dtype=torch.bool), X, torch.ones_like(X) * -1000)
        y = torch.where((torch.ones_like(y) * mask[:, None]).to(dtype=torch.bool), y, torch.ones_like(y) * -1000)
        var = torch.where((torch.ones_like(var) * mask[:, None]).to(dtype=torch.bool), var, torch.ones_like(var) * -1000)

        if len(self.surface_threshold) == 1:
            mask = y.squeeze() >= self.surface_threshold[0]
        else:
            min_mask = y.squeeze() >= self.surface_threshold[0]
            max_mask = y.squeeze() <= self.surface_threshold[1]
            mask = torch.logical_and(min_mask, max_mask)

        # X = torch.where((torch.ones_like(X) * mask[:, None]).to(dtype=torch.bool), X, torch.ones_like(X) * -1000)
        y = torch.where((torch.ones_like(y) * mask[:, None]).to(dtype=torch.bool), y, torch.ones_like(y) * -1000)
        var = torch.where((torch.ones_like(var) * mask[:, None]).to(dtype=torch.bool), var, torch.ones_like(var) * -1000)

        var_mask = var.squeeze(-1) <= self.variance_threshold
        # X = torch.where((torch.ones_like(X) * var_mask[:, None]).to(dtype=torch.bool), X, torch.ones_like(X) * -1000)
        y = torch.where((torch.ones_like(y) * var_mask[:, None]).to(dtype=torch.bool), y, torch.ones_like(y) * -1000)
        var = torch.where((torch.ones_like(var) * var_mask[:, None]).to(dtype=torch.bool), var, torch.ones_like(var) * -1000)

        return X, y, var

    def _plot_velocity_volumetric(self, Xqs, yqs, fig, row, col, plot_args=None):
        """
        # generic method for any plot
        :param Xqs: filtered Nx3 position
        :param yqs:  filtered N values
        :param fig:
        :param row:
        :param col:print("Number of points after filtering: ", Xq_mv.shape[0])
        :param plot_args: symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
        """

        print(" Plotting row {}, col {}".format(row, col))
        fname_in = "./datasets/kyle_ransalu/5_airsim/5_airsim1/5_airsim1_vel_train_normalized_infilled"
        prefilled_X = pd.read_csv(fname_in + '.csv', delimiter=',').to_numpy()[:2542,1:4]
        mask = np.sum(euclidean_distances(Xqs, prefilled_X) <= 0.3, axis=1) >= 1
        # Xqs = torch.where((torch.ones_like(Xqs) * mask[:, None]).to(dtype=torch.bool), Xqs, torch.ones_like(Xqs) * -1000)
        yqs = torch.where((torch.ones_like(yqs) * mask[:, None]).to(dtype=torch.bool), yqs, torch.ones_like(yqs) * -1000)

        # marker and colorbar arguments
        if plot_args is None:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = 'square', 8, 0.2, False, yqs[:,0].min(), yqs[:,0].max()
        else:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = plot_args
        if cbar_x_pos is not False:
            colorbar = dict(x=cbar_x_pos,
                            len=1,
                            y=0.5
                        )
        else:
            colorbar = dict()

        colorbar["tickfont"] = dict(size=18)

        fig.add_trace(
            go.Volume(
                x=Xqs[:, 0],
                y=Xqs[:, 1],
                z=Xqs[:, 2],
                isomin=-7,
                isomax=7,
                value=yqs,
                opacity=0.05,
                surface_count=40,
                colorscale="Jet",
                opacityscale=[[0, 0], [self.surface_threshold[0], 0], [1, 1]],
                colorbar=colorbar,
                # cmax=1,
                # cmin=self.surface_threshold[0],
            ),
            row=1,
            col=2
        )

    def _plot_velocity_scatter(self, Xqs, yqs, fig, row, col, plot_args=None):
        """
        # generic method for any plot
        :param Xqs: filtered Nx3 position
        :param yqs:  filtered N values
        :param fig:
        :param row:
        :param col:print("Number of points after filtering: ", Xq_mv.shape[0])
        :param plot_args: symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
        """

        print(" Plotting row {}, col {}".format(row, col))
        # marker and colorbar arguments
        if plot_args is None:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = 'square', 8, 0.2, False, yqs[:,0].min(), yqs[:,0].max()
        else:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = plot_args
        if cbar_x_pos is not False:
            colorbar = dict(x=cbar_x_pos,
                            len=1,
                            y=0.5
                        )
        else:
            colorbar = dict()

        colorbar["tickfont"] = dict(size=18)

        # plot
        fig.add_trace(
            go.Scatter3d(
                x=Xqs[:,0],
                y=Xqs[:,1],
                z=Xqs[:,2],
                mode='markers',
                marker=dict(
                    color=yqs[:,0],
                    colorscale="Jet",
                    cmax=cbar_max,
                    cmin=cbar_min,
                    colorbar=colorbar,
                    opacity=opacity,
                    symbol=symbol,
                    size=size
                ),
            ),
            row=row,
            col=col
        )

    def _plot_velocity_1by3(self, X, y_vy, Xq_mv, mean_y, var_y, i):
        """
        # This plot is good for radar data
        :param X: ground truth positions
        :param y_vy: ground truth y velocity
        :param Xq_mv: query X positions
        :param mean_y: predicted y velocity mean
        :param i: ith frame
        """
        print(" Plotting 1x3 subplots")

        # setup plot
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],]
        titles = ["Ground truth", "Prediction (mean)", "Predictions (variance)"]
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=specs,
            subplot_titles=titles
        )

        # filter by surface threshold
        print(" Surface_thresh: ", self.surface_threshold)
        print(" Number of points before filtering: {}".format(Xq_mv.shape[0]))

        if self.plot_volumetric:
            Xq_mv, mean_y, var_y = self._filter_predictions_velocity_where(Xq_mv, mean_y, var_y)
        else:
            Xq_mv, mean_y, var_y = self._filter_predictions_velocity(Xq_mv, mean_y, var_y)

        print(" Number of points after filtering: {}".format(Xq_mv.shape[0]))

        # set colorbar
        cbar_min = min(mean_y.min().item(), y_vy.min().item())
        cbar_max = max(mean_y.max().item(), y_vy.max().item())
        print(f"mean_y.min().item(): {mean_y.min().item()}, y_vy.min().item(): {y_vy.min().item()}")
        print(f"mean_y.max().item(): {mean_y.max().item()}, y_vy.max().item(): {y_vy.max().item()}")
        # fig.update_layout(coloraxis={'colorscale':'Jet', "cmin":cbar_min, "cmax":max_c}) # global colrobar

        # plot
        # plot_args - symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
        plot_setting = 5
        if plot_setting == 1: #for 1x3, scatter, shared colorbar
            plot_args_data =      ['circle', 5, 0.7, 0.3, cbar_min, cbar_max]
            plot_args_pred_mean = ['circle', 5, 0.7, 0.6, cbar_min, cbar_max] #opacity=0.1
            plot_args_pred_var = ['circle', 5, 0.7, 0.9, None, None] #opacity=0.1
        elif plot_setting == 2: #for 1x3, scatter, separate axis
            plot_args_data =      ['circle', 5, 0.7, 0.3, None, None]
            plot_args_pred_mean = ['circle', 5, 0.7, 0.6, None, None] #opacity=0.1
            plot_args_pred_var = ['circle', 5, 0.7, 0.9, None, None] #opacity=0.1
        elif plot_setting == 3: #for 1x3 query slice, shared colobar
            plot_args_data =      ['circle', 5, 0.7, 0.25, cbar_min, cbar_max]
            plot_args_pred_mean = ['square', 5, 0.7, 0.63, cbar_min, cbar_max] #opacity=0.1
            plot_args_pred_var = ['square', 5, 0.7, 0.95, None, None] #opacity=0.1
        elif plot_setting == 4:  # for 1x3 query slice, separate colobar
            plot_args_data = ['circle', 5, 0.7, 0.25, None, None]
            plot_args_pred_mean = ['square', 5, 0.7, 0.63, None, None]
            plot_args_pred_var = ['square', 5, 0.7, 0.95, None, None]
        elif plot_setting == 5: #for 1x3 query everywhere, shared colobar
            plot_args_data =      ['circle', 5, 0.7, 0.265, cbar_min, cbar_max]
            # plot_args_data =      ['circle', 1.5, 0.7, 0.265, cbar_min, cbar_max]
            plot_args_pred_mean = ['square', 2.5, 0.4, 0.63, cbar_min, cbar_max] #opacity=0.1
            plot_args_pred_var = ['square', 2.5, 0.4, 0.975, 0, None] #opacity=0.1
        elif plot_setting == 6: #for 1x3 query everywhere, sperate colorbar
            plot_args_data =      ['circle', 3, 0.7, 0.3, None, None]
            plot_args_pred_mean = ['square', 3, 0.3, 0.6, None, None] #opacity=0.1
            plot_args_pred_var = ['square', 3, 0.3, 0.9, None, None] #opacity=0.1
        else:
            pass

        if self.plot_volumetric:
            self._plot_velocity_volumetric(X.float(), y_vy, fig, 1, 1, plot_args_data)
            self._plot_velocity_volumetric(Xq_mv.float(), mean_y.float(), fig, 1, 2, plot_args_pred_mean)
            self._plot_velocity_volumetric(Xq_mv.float(), var_y.float(), fig, 1, 3, plot_args_pred_var)
        else:
            self._plot_velocity_scatter(X.float(), y_vy, fig, 1, 1, plot_args_data)
            self._plot_velocity_scatter(Xq_mv.float(), mean_y.float(), fig, 1, 2, plot_args_pred_mean)
            self._plot_velocity_scatter(Xq_mv.float(), var_y.float(), fig, 1, 3, plot_args_pred_var)

        # update camera
        camera = dict(
            eye=dict(x=2.25, y=-2.25, z=1.25)
            # eye=dict(x=-2.25, y=-2.25, z=1.25)
            # eye=dict(x=-4, y=0.2, z=0.5)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera
        fig.layout.scene3.camera = camera

        # update plot settings
        layout = dict(xaxis=dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]], ),
                      yaxis=dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]], ),
                      zaxis=dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]], ),
                      aspectmode="manual",
                      aspectratio=dict(x=2, y=2, z=2))
        fig.update_layout(scene1=layout, scene2=layout, scene3=layout, font=dict(size=15))

        plots_dir = os.path.abspath("./plots/velocity")
        if not os.path.isdir(plots_dir):
            print(f"Creating \"{plots_dir}\"...")
            os.makedirs(plots_dir)

        fig.update_layout(title='{}_velocity_frame{}'.format(self.plot_title, i), height=450)
        filename = os.path.abspath('./plots/velocity/{}_frame{}.html'.format(self.plot_title, i))
        plotly.offline.plot(fig, filename=filename, auto_open=False)
        print(' Plot saved as ' + filename)

        pdf_filename = os.path.abspath('./plots/velocity/{}_frame{}.pdf'.format(self.plot_title, i))
        fig.write_image(pdf_filename, width=1500, height=450)
        print(' Plot also saved as ' + pdf_filename)

        svg_filename = os.path.abspath('./plots/velocity/{}_frame{}.svg'.format(self.plot_title, i))
        fig.write_image(svg_filename, width=1500, height=450)
        print(' Plot also saved as ' + svg_filename)

        png_filename = os.path.abspath('./plots/velocity/{}_frame{}.png'.format(self.plot_title, i))
        fig.write_image(png_filename, width=1500, height=450)
        print(' Plot also saved as ' + png_filename)


    def plot_velocity_frame(self, X, y_vx, y_vy, y_vz, Xq_mv, mean_x, var_x, mean_y, var_y, mean_z, var_z, i):
        time1 = time.time()

        if self.plot_axis == "x":
            self._plot_velocity_1by3(X, y_vx, Xq_mv, mean_x, var_x, i)
        elif self.plot_axis == "y":
            self._plot_velocity_1by3(X, y_vy, Xq_mv, mean_y, var_y, i)
        elif self.plot_axis == "z":
            self._plot_velocity_1by3(X, y_vz, Xq_mv, mean_z, var_z, i)
        else:
            raise ValueError("Unsupported plot axis \"{self.plot_axis}\"")

        print(' Total plotting time=%2f s' % (time.time()-time1))
