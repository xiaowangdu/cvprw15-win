import os
import numpy as np
from sklearn.model_selection import train_test_split

#we use method 'hold-out' generate test set.
split_test = 1

def list_dir(path, file_list):
    if os.path.isdir(path):
        if path.split("/")[-1] != ".git":
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                list_dir(file_path, file_list) 
    else:
        if path.endswith(".jpg"):
            file_list.append(os.path.normpath(path))
        
def get_labels(file_list):
    labels = []
    for file in file_list:
        label = os.path.basename(file).split('_')[0]
        labels.append(label)
        
    return labels
 
def save_as_listfile(target_file, files, labels):
    f = open(target_file,'w')
    for (file, label) in zip(files, labels):
        f.write('%s %s\n'%(file.replace('\\', '/'), label))
    f.close()
 
def main():
    filenames = [] 
    list_dir('256_ObjectCategories', filenames)
    labels = get_labels(filenames)
    if split_test:
        X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.20, random_state=0) 
        print(len(X_train), len(X_test), len(y_train), len(y_test))
        save_as_listfile('train.txt', X_train, y_train)
        save_as_listfile('val.txt', X_test, y_test)
    else:
        save_as_listfile('train.txt', filenames, labels)
        
        
if __name__ == '__main__':
    main()