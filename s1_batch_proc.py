import numpy as np 
from mylib_feature_proc import *
from mylib_io import *
from mylib_commons import *

def get_filenames(path, format="*", no_prefix=False):
    import glob
    list_filenames = sorted(glob.glob(path+'/'+format))
    if no_prefix:
        list_filenames = [ name.split('/')[-1]
            for name in list_filenames]
    return list_filenames  # Path to file wrt current folder

def load_classes(data_folder):
    classes = get_filenames(data_folder, no_prefix=True)
    classes = [c for c in classes if ('.' not in classes)]
    print(classes)
    return classes

if __name__=="__main__":
    data_folder = "data_src"
    train_X = []
    train_Y = []
    classes = load_classes(data_folder)
    for yi, label in enumerate(classes):
        folder = data_folder + "/" + label + "/"
        fnames = get_filenames(folder)
        for fn in fnames:
            if fn[-4:] != ".wav": continue
            data, sample_rate = read_audio(fn)

            # Preprocess data
            data = remove_data_prefix(data, sample_rate)
            print("File: {}, data: {}".format(fn, data.shape))

            datas = [data]
            NUM_AUGMENTS = 0 # Data augment reduce the performance
            for i in range(NUM_AUGMENTS):
                datas.append(data_augment(data, sample_rate))

            # Compute features
            for ith_data, data in enumerate(datas):
                xi = data_to_features(data, sample_rate)
                # print("  {}th data, len = {}".format(ith_data, data.size))

                # Print and plot
                if 0:
                    fname_jpg = change_suffix(fn, index=ith_data, new_suffix="jpg")
                    cv2.imwrite(fname_jpg, cv2_image_f2i(xi))
                
                xi = np.ravel(xi)
                # Store data
                train_X.append(xi)
                train_Y.append(yi)
            # print("")
            continue

    train_X, train_Y = np.array(train_X), np.array(train_Y, dtype=np.int)

    # Print info
    print("train_X.shape = {}".format(train_X.shape))
    print("train_Y.shape = {}".format(train_Y.shape))

    # Save
    np.savetxt("train_X.csv", train_X)
    np.savetxt("train_Y.csv", train_Y)
    write_list("classes.csv", classes)