

import types 
import utils.lib_datasets as lib_datasets

args = types.SimpleNamespace()
args.data_folder = "" # TODO
args.classes_txt = "" # TODO

# Get data's filenames and labels
files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

all_data = lib_datasets.AudioDataset(
    files_name=files_name, files_label=files_label, transform=None)

for i in range(len(all_data)):
    audio = all_data.get_audio(i)
    lens = audio.get_len_s()
    if lens < 0.5: # Print audios with a short length
        print("len={}, filename={}".format(
            lens, audio.filename))
        
exit("complete debug: audio's length")
