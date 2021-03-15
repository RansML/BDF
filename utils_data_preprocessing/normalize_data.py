import numpy as np
import pandas as pd

def normalize_minmax(fname_in, fname_out, min_val=0, max_val=1, columns=None):
    data = pd.read_csv(fname_in + '.csv', delimiter=',')

    if columns == None:
        x = data
    else:
        print(data.columns.tolist())
        x = data[columns]

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    x_normalized = min_val + (x - x_min)/(x_max - x_min)*(max_val - min_val)

    if columns == None:
        data = x_normalized
    else:
        data[columns] = x_normalized

    pd.DataFrame(data).to_csv(fname_out + '.csv', index=False)

    return data

if __name__ == "__main__":
    # fname_in = "./datasets/kyle_ransalu/1_toy/1_toy1_vel_train"
    # fname_out = fname_in + '_normalized'
    # data = normalize_minmax(fname_in, fname_out, min_val=-1, max_val=1, columns=[' X', ' Y', ' Z'])
    # print(data.min(axis=0))
    # print(data.max(axis=0))

    import os
    from glob import glob
    from tqdm import tqdm

    FORMATTED_FILES_DIR = "./datasets/kyle_ransalu/3_astyx/3_astyx1/formatted"
    NORMALIZED_FILES_FILES_DIR = "./datasets/kyle_ransalu/3_astyx/3_astyx1/normalized2"

    if not os.path.isdir(NORMALIZED_FILES_FILES_DIR):
        os.makedirs(NORMALIZED_FILES_FILES_DIR)

    formatted_files = glob(os.path.join(FORMATTED_FILES_DIR, "*.csv"))
    for formatted_file in tqdm(formatted_files, total=len(formatted_files)):
        name = formatted_file.split("/")[-1].split(".")[0]
        fname_in = os.path.join(FORMATTED_FILES_DIR, name)
        fname_out = os.path.join(NORMALIZED_FILES_FILES_DIR, name)
        data = normalize_minmax(fname_in, fname_out, min_val=-1, max_val=1, columns=['X', 'Y', 'Z'])
