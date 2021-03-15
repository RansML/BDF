import numpy as np
import os
from glob import glob
from tqdm import tqdm
from nuscenes.utils.data_classes import RadarPointCloud

PCD_DIRS = "../../v1.0-mini/samples/RADAR_FRONT"
# FORMATTED_DATA_DIR = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/Bayesian-Dynamics/datasets/nuscenes/samples/formatted/RADAR_FRONT"
# NORMALIZED_DATA_DIR = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/Bayesian-Dynamics/datasets/nuscenes/samples/normalized/RADAR_FRONT"
FORMATTED_DATA_DIR = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/SpaTUn/datasets/kyle_ransalu/4_nuscenes/4_nuscenes1/samples/formatted/RADAR_FRONT"
NORMALIZED_DATA_DIR = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/SpaTUn/datasets/kyle_ransalu/4_nuscenes/4_nuscenes1/samples/formatted/RADAR_FRONT"

def format_all_files(pcd_dirs, formatted_data_dir_path, normalized_data_dir_path):
    radar_files = glob(os.path.join(pcd_dirs, "*.pcd"))
    for radar_file in tqdm(radar_files, total=len(radar_files), desc="Formatting files"):
        radar_name = get_name_from_radar_file(radar_file)
        formatted_data_path = os.path.join(formatted_data_dir_path, radar_name + ".csv")
        normalized_data_path = os.path.join(normalized_data_dir_path, radar_name + ".csv")
        run_format(radar_file, formatted_data_path, normalized_data_path)
    print("Done")


def get_name_from_radar_file(radar_file):
    return radar_file.split(".")[-2].split("/")[-1]


def run_format(pcd_file, formatted_data_path, normalized_data_path):
    pcd = RadarPointCloud.from_file(pcd_file)
    radar_points = pcd.points.T
    print("radar_points.shape:", radar_points.shape)

    positions = radar_points[:, 0:3]
    print("positions.shape:", positions.shape)
    print("positions:", positions)
    velocities = radar_points[:, 6:8]
    print("velocities.shape:", velocities.shape)
    print("velocities:", velocities)

    formatted_data = np.concatenate((np.zeros((radar_points.shape[0], 1)), positions, velocities, np.zeros((radar_points.shape[0], 1))), axis=1)

    print("formatted_data.shape:", formatted_data.shape)

    header = "t,X,Y,Z,V_X,V_Y,V_Z"

    formatted_data_dir = "/" + os.path.join(*formatted_data_path.split("/")[:-1])
    print("formatted_data_dir:", formatted_data_dir)
    if not os.path.isdir(formatted_data_dir):
        os.makedirs(formatted_data_dir)

    np.savetxt(formatted_data_path, formatted_data, delimiter=",", header=header, comments="")

    # Normalize data
    for col in range(1, formatted_data.shape[1]):
        min_val = np.amin(formatted_data[:, col])
        max_val = np.amax(formatted_data[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)

        if min_val != 0 or max_val != 0: # Leave columns with only 0s alone
            formatted_data[:, col] -= min_val
            formatted_data[:, col] /= (max_val - min_val)
            formatted_data[:, col] *= 2
            formatted_data[:, col] -= 1

    for col in range(1, formatted_data.shape[1]):
        min_val = np.amin(formatted_data[:, col])
        max_val = np.amax(formatted_data[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)


    normalized_data_dir = "/" + os.path.join(*normalized_data_path.split("/")[:-1])
    print("normalized_data_dir:", normalized_data_dir)
    if not os.path.isdir(normalized_data_dir):
        os.makedirs(normalized_data_dir)

    np.savetxt(normalized_data_path, formatted_data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    # run_format(PCD_PATH, FORMATTED_DATA_PATH, NORMALIZED_DATA_PATH)
    format_all_files(PCD_DIRS, FORMATTED_DATA_DIR, NORMALIZED_DATA_DIR)
