import numpy as np

DATA_PATH = "./datasets/toy_velocity/toy_velocity_1.csv"
# DATA_PATH = "./datasets/velocity1/radar_carla_test1_frame_250.csv"

def add_timesteps(data_path):
    new_file_contents = ""

    i = -1
    with open(data_path,  "r") as f:
        for line in f:
            if i < 0:
                new_file_contents += "t," + line.replace(" ", "")
            else:
                # new_file_contents += "{},".format(i) + line.replace(" ", "")
                new_file_contents += "{},".format(0) + line.replace(" ", "")
            i += 1

    with open(data_path,  "w") as f:
        f.write(new_file_contents)

def get_header(data_path):
    with open(data_path, "r") as f:
        line = f.readline()
        header = line.replace("\n", "").split(",")
        return header

if __name__ == "__main__":
    add_timesteps(DATA_PATH)
