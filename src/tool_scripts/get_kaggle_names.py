# print the filenames in a folder

import os 
names = [name for name in os.listdir("kaggle/") if os.path.isdir(name)] 
for name in names:
    print(name)
