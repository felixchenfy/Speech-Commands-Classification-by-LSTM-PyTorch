import numpy as np 
from mylib.mylib_feature_proc import *
from mylib.mylib_io import *
from mylib.mylib_commons import *

import sys, os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


def get_filenames(path, format="*", no_prefix=False):
    import glob
    list_filenames = sorted(glob.glob(path+'/'+format))
    if no_prefix:
        list_filenames = [ name.split('/')[-1]
            for name in list_filenames]
    return list_filenames  # Path to file wrt current folder

def load_classes(data_folder):
    classes = get_filenames(data_folder, no_prefix=True)
    classes = [c for c in classes if ('.' not in c)]
    print(classes)
    return classes

if __name__=="__main__":
    data_folder =  "data_src"
    train_X = []
    train_Y = []
    classes = load_classes(data_folder)
    for yi, label in enumerate(classes):
        folder = data_folder + "/" + label + "/"
        fnames = get_filenames(folder)
        for ith_frame, fn in enumerate(fnames):

            # Check file
            if fn[-4:] != ".wav": continue
            # if (ith_frame+1) % 3 == 0: continue

            # Read .wav file
            data, sample_rate = read_audio(fn)

            # Rename and save the audio to file 
            if ith_frame == 0:
                write_audio(data_folder + "/" + label + ".wav", data, sample_rate)
            
            # Preprocess data
            data = remove_data_prefix(data, sample_rate)
            datas = [data]

            # Data augmentation
            NUM_AUGMENTS = 0 # Data augment reduce the performance
            for i in range(NUM_AUGMENTS):
                datas.append(data_augment(data, sample_rate))

            # Compute features
            for ith_aug, data in enumerate(datas):
                xi = data_to_features(data, sample_rate)


                # Print data info
                if ith_aug == 0:
                    print("Class = {}, index = {}/{}, data = {}, feature = {}, filename = {}, ".format(
                        yi, ith_frame, len(fnames), data.shape, xi.shape, fn.split('/')[-1]),
                        end='\n')

                # Print and plot
                if 0:
                    fname_jpg = change_suffix(fn, index=ith_aug, new_suffix="jpg")
                    cv2.imwrite(fname_jpg, cv2_image_f2i(xi))
                
                xi = np.ravel(xi)

                # Store data
                train_X.append(xi.tolist())
                train_Y.append(yi)
            # print("")
            continue


    # Print info
    print("train_X = List[List[]] = ({}, {})".format(len(train_X), len(train_X[0])))
    print("train_Y = List         = {}".format(len(train_Y)))

    # Save
    write_list("train_X.csv", train_X)
    write_list("train_Y.csv", train_Y)
    write_list("classes.csv", classes)

    # train_X, train_Y = np.array(train_X), np.array(train_Y, dtype=np.int)

    # # Print info
    # print("train_X.shape = {}".format(train_X.shape))
    # print("train_Y.shape = {}".format(train_Y.shape))

    # # Save
    # np.savetxt("train_X.csv", train_X)
    # np.savetxt("train_Y.csv", train_Y)
    # write_list("classes.csv", classes)