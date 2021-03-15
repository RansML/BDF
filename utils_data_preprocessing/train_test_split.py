import pandas as pd
import numpy as np
import os

def split(data, propotion):
    """
    :param data: data matrix
    :param propotion: as a fraction e.g. 0.2 for 20% test
    :return: train set, test set
    """
    N = data.shape[0]
    all_indx = np.arange(N)
    test_indx = np.random.choice(all_indx, np.int(N*propotion), replace=False)
    train_indx = np.in1d(all_indx, test_indx, invert=True)
    print(' Orig size={}, Train size={}, Test size={}'.format(data.shape[0], train_indx.shape[0], test_indx.shape[0]))

    return data[train_indx, :], data[test_indx, :]

def split_n_save(fn, propotion=0.2):
    """
    :param fn: file name of the raw dataset
    :param propotion: as a fraction e.g. 0.2 for 20% test
    """
    print(' Train-test {} splitting '.format(propotion) + fn + '.csv...')
    data = pd.read_csv(fn+'.csv', delimiter=',')
    train_set, test_set = split(data.values, propotion)

    header = data.columns.values
    header = ', '.join(header)
    with open(fn + '_train.csv', 'w') as f_handle:  # try 'a'
        np.savetxt(f_handle, train_set, delimiter=',', header=header, comments='')
    with open(fn + '_test.csv', 'w') as f_handle:  # try 'a'
        np.savetxt(f_handle, test_set, delimiter=',', header=header, comments='')

    print(' Train-test datasets saved.')

if __name__ == "__main__":
    # files = \
    #     ['1_toy/1_toy1_vel',
    #      '2_carla/2_carla1_frame_250',
    #      '3_astyx/3_astyx1/normalized/000002',
    #      '4_nuscenes/4_nuscenes1/samples/normalized/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991',
    #      '5_airsim/5_airsim1/5_airsim1_pos',
    #      '5_airsim/5_airsim1/5_airsim1_vel',
    #      '6_jfk/6_jfk_partial1_pos',
    #      '6_jfk/6_jfk_partial1_vel'
    #      ]

    # for f in files:
    #     split_n_save('../datasets/kyle_ransalu/' + f, 0.2)


    file_base = "3_astyx/3_astyx1/normalized/{:06d}"
    file_idxs_start = 0
    file_idx_stop = 9

    for i in range(file_idxs_start, file_idx_stop + 1):
        f = file_base.format(i)
        print(f)
        split_n_save('../datasets/kyle_ransalu/' + f, 0.2)
