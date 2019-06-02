


# =============================================================
# =============================================================
# =============================================================
# =============================================================
# =============================================================

# Part of train.py
if 0: # DEBUG: use only a subset of all data
    GAP = 10
    files_name = files_name[::GAP]
    files_label = files_label[::GAP]
    args.num_epochs = 5
    
if 0: # DEBUG: check length of each data
    all_data = lib_datasets.AudioDataset(
        files_name=files_name, files_label=files_label, transform=None)
    for i in range(len(all_data)):
        audio = all_data.get_audio(i)
        lens = audio.get_len_s()
        if lens < 0.5:
            print("len={}, filename={}".format(
                lens, audio.filename))
    exit("complete debug: audio's length")


# ------------------------------------------------------------------------
# if 0: # Eval the model on some test set
    
    
#     if 1: # Test dataset same as eval set
#         import copy 
#         test_dataset = copy.deepcopy(eval_dataset)
        
#     elif 0: # DEBUG: test data from separate folder
#         IF_TRAIN_MODEL = False
#         folder = "data/data_test"
        
#         files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
#             folder, args.classes_txt)
        
#         test_dataset = lib_datasets.AudioDataset(
#             files_name=files_name, files_label=files_label, transform=None)
        
#         print(f"Test on {len(test_dataset)} samples from {folder}")


#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

#     if IF_LOAD_FROM_PRETRAINED:
#         model.load_state_dict(torch.load(LOAD_PRETRAINED_PATH))

#     model.eval()    

#     with torch.no_grad():
#         lib_rnn.evaluate_model(model, test_loader)




# =============================================================
# =============================================================
# =============================================================
# =============================================================
# =============================================================







# =============================================================
# =============================================================
# =============================================================
# =============================================================
# =============================================================





# =============================================================
# =============================================================
# =============================================================
# =============================================================
# =============================================================