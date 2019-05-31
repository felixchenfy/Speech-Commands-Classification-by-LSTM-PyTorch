import numpy as np 
from utils.lib_feature_proc import *
from utils.lib_io import *
from utils.lib_commons import *
from utils.lib_plot import *

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

    def extract_features_from_raw_data(
        data_folder, fname_data_X, fname_data_Y,
        if_data_aug):
        data_X = []
        data_Y = []
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
                # play_audio(data=data, sample_rate=sample_rate)

                # Rename and save the audio to file 
                if ith_frame == 0:
                    write_audio(data_folder + "/" + label + ".wav", data, sample_rate)
                
                # Preprocess data
                # data = remove_silent_prefix(data, sample_rate)
                datas = [data]

                # Data augmentation
                if if_data_aug:
                    NUM_AUGMENTS = 3 
                else:
                    NUM_AUGMENTS = 0
                for i in range(NUM_AUGMENTS):
                    datas.append(data_augment(data, sample_rate))

                # Compute features
                for ith_aug, data in enumerate(datas):
                    xi = data_to_features(data, sample_rate)


                    # Print data info
                    if ith_aug == 0:
                        print("Class = {}, index = {}/{}, data = {}, maxmin = ({:.2f}, {:.2f})， feature = {}, filename = {}, ".format(
                            yi, ith_frame, len(fnames), data.shape, data.max(), data.min(), xi.shape, fn.split('/')[-1]),
                            end='\n')

                    # Print and plot
                    if 0:
                        plt.cla()
                        fname_jpg = change_suffix(fn, index=ith_aug, new_suffix="jpg")
                        plot_audio(data, sample_rate)
                        plt.savefig(fname_jpg)
                        # cv2.imwrite(fname_jpg, cv2_image_f2i(xi))
                    
                    xi = np.ravel(xi)

                    # Store data
                    data_X.append(xi.tolist())
                    data_Y.append(yi)
                # print("")
                continue


        # Print info
        print("data/data_X = List[List[]] = ({}, {})".format(len(data_X), len(data_X[0])))
        print("data/data_Y = List         = {}".format(len(data_Y)))

        # Save
        write_list(fname_data_X, data_X)
        write_list(fname_data_Y, data_Y)
        write_list("classes.csv", classes)

    plt.figure(figsize=(15, 6))
    
    data_folder =  "data/data_train"
    fname_data_X = "data/train_X.csv"
    fname_data_Y = "data/train_Y.csv"
    extract_features_from_raw_data(data_folder, fname_data_X, fname_data_Y, if_data_aug=True)


    # data_folder =  "data/data_test"
    # fname_data_X = "data/test_X.csv"
    # fname_data_Y = "data/test_Y.csv"
    # extract_features_from_raw_data(data_folder, fname_data_X, fname_data_Y, if_data_aug=False)


    # data_folder =  "data/data_test2"
    # fname_data_X = "data/test2_X.csv"
    # fname_data_Y = "data/test2_Y.csv"
    # extract_features_from_raw_data(data_folder, fname_data_X, fname_data_Y, if_data_aug=True)