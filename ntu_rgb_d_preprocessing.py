
# ======================================================= #
#
# Preprocessing NTU-RGB-D dataset
#   - remove noisy skeletons with aspect ration greater than 0.85
#   - remove other skeletons but the two with highest motion
#   - sample F frames tactically to use to calculate features
#   - normalize data for K frames (center of mass, std,..)
#   - add more hand-crafted features like distance between two skeletons,...
#
# ======================================================= #
import os
from os.path import join
import numpy as np

# path of .npy files
npy_path = "dataset-NTU-RGB-D/nturgb+d_npy/"
# get .npy files
npy_file_names = [file for file in os.listdir(npy_path) if file.endswith('skeleton.npy')]

# get .clean.npy files
existed_clean_files = [file.replace(".clean","") for file in os.listdir(npy_path) if file.endswith('clean.npy')]

def remove_noisy_mistaken_objects():
    #
    # remove noisy skeletons with aspect ratio (width/height) greater than 0.8
    #
    for name in npy_file_names:
        # check if the file is processed
        if name in existed_clean_files:
            print ("The file already existed. Ignore!")
            continue

        # full path
        full_path = join(npy_path, name)
        # load data from file as a dictionary
        bodysdata = np.load(full_path,allow_pickle=True).item()

        nbodys = max(bodysdata['nbodys'])

        # check max number of bodies of every frames
        # if equal to 1, ignore the file
        if nbodys < 2:
            continue

        # save actual bodys and ignore noisy bodys
        actual_bodys = []
        for body in range(nbodys):
            # get i-th body rgb data (ndarray type)
            rgb_body_data = bodysdata['rgb_bodys'][body]

            nframes = len(rgb_body_data)
            for j in range(nframes):
                # ignore zero frames
                if not np.any(rgb_body_data[j]):
                    continue
                # calculate spread of X and Y
                minX, maxX = min(rgb_body_data[j][:,0]),max(rgb_body_data[j][:,0])
                minY, maxY = min(rgb_body_data[j][:,1]), max(rgb_body_data[j][:,1])

                # check aspect ration
                if (maxX-minX)/(maxY-minY) > 0.85:
                    pass
                else:
                    # save this actual body index
                    actual_bodys.append(body)
                break

        if len(actual_bodys) < nbodys:
            print('Remove noisy body from {}'.format(name))
            # save only actual bodys
            bodysdata['rgb_bodys'] = bodysdata['rgb_bodys'][actual_bodys]

            # update nbodys
            bodysdata['nbodys'] = np.subtract(bodysdata['nbodys'], nbodys - len(actual_bodys))

            # save back data to .npy file
            np.save(full_path, bodysdata)

    pass

def remove_slow_motion_bodys():
    #
    # remove slow-motion bodys, and keep only two highest-motion bodys
    # the main actor will be at index of 0
    #
    total_ignore_files = 0
    for name in npy_file_names:
        # check if the file is processed
        if name in existed_clean_files:
            print ("The file already existed. Ignore!")
            continue
        # full path
        full_path = join(npy_path, name)
        # load data from file as a dictionary
        bodysdata = np.load(full_path,allow_pickle=True).item()

        # list of list of [body_index,body_motion]
        bodys_motions = []

        nbodys = max(bodysdata['nbodys'])

        # there's only one body, ignore this file
        if nbodys <= 1:
            total_ignore_files += 1
            print('{} has less than 2 bodys. Ignore!'.format(name))
            continue
        # if there are only 2 bodys, do nothing and save to .clean file
        elif nbodys == 2:
            np.save(full_path[:-4] + '.clean', bodysdata)
            continue

        # if there are more than 2 bodys, compute motion
        for body in range(nbodys):
            # get i-th body rgb data (ndarray type)
            rgb_body_data = bodysdata['rgb_bodys'][body]

            # get not empty frame mask
            not_empty_frame_mask = [frame.any() for frame in rgb_body_data]

            accumulative_motion = [[0,0]]*rgb_body_data.shape[1]
            for j in range(1,len(rgb_body_data[not_empty_frame_mask])):
                accumulative_motion += np.abs(rgb_body_data[not_empty_frame_mask][j] - rgb_body_data[
                    not_empty_frame_mask][j-1])

            # sum up motion and save to tuple
            bodys_motions.append([body, np.sum(accumulative_motion)])

        # sort list of bodys_motions order by descending motion
        bodys_motions = sorted(bodys_motions,key=lambda x: x[1],reverse=True)

        # remove other bodys and keep only two highest-motion bodys
        bodysdata['rgb_bodys'] = bodysdata['rgb_bodys'][[item[0] for item in bodys_motions[:2]]]

        # update nbodys equal to 2
        bodysdata['nbodys'] = len(bodysdata['nbodys'])*[2]

        # save back data to .clean file
        np.save(full_path[:-4] + '.clean', bodysdata)

    print('Ignore {} files in total.'.format(total_ignore_files))

    pass


if __name__ == "__main__":
    remove_noisy_mistaken_objects()

    remove_slow_motion_bodys()

    pass