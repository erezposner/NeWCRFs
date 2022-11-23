import random

from scripts.load_data import load_data_from_demulator_folder

overfit = False

def separate_train_validate_test(l,percentage):
    sz = len(l)
    shuffled_l = l.copy()
    # random.shuffle(shuffled_l)
    l1 = shuffled_l[:int(sz*percentage[0])]  # first 80% of shuffled list
    l2 = shuffled_l[int(sz*percentage[0]):(int(sz*percentage[0])+int(sz*percentage[1]))]  # first 80% of shuffled list
    l3 = shuffled_l[(int(sz*percentage[0])+int(sz*percentage[1])):]  # first 80% of shuffled list

    # if overfit:
    #     return l2,l2
    return l1,l2,l3
import glob
from pathlib import Path

path = '/mnt/data/GIVideoFrame/synthetic/demulator/Nov08_2022'
depthmaps, colors, poses, D, K = load_data_from_demulator_folder(path, number_of_point_clouds_to_plot=10, voxel_size=0.005)


train_val_files_list = []
test_files_list = []
percentage = [0.7,0.2,0.1]
for i,_ in enumerate(depthmaps):
    # print(folder)
    # remove linebreak from a current name
    # linebreak is the last character of each line
    train_val_files_list.append(str(colors[i]) + ' ' + str(depthmaps[i]) + '\n')


train, val, test = separate_train_validate_test(train_val_files_list, percentage)

with open(rf'data_splits/{Path(path).name}_demulator_train_files_with_gt.txt', 'w') as fp:
    fp.writelines(train)
with open(rf'data_splits/{Path(path).name}_demulator_valid_files_with_gt.txt', 'w') as fp:
    fp.writelines(val)
with open(rf'data_splits/{Path(path).name}_demulator_test_files_with_gt.txt', 'w') as fp:
    fp.writelines(test)

# with open(r'data_splits/splits_colsim.txt', 'r') as fp:
#         filenames = fp.readlines()
#
# print('read')