import sys, os

def load(file_list_path, root_dir = ''):
    fin = open(file_list_path)
    images_list = []
    for line in fin.readlines():
        t = line.rstrip()
        im_name = os.path.join(root_dir, t)
        images_list.append(im_name)
    fin.close()
    return images_list
  
def save_as_listfile(target_file, files, labels = None):
    f = open(target_file,'w')
    if labels is not None:
        for (file, label) in zip(files, labels):
            f.write('%s %s\n'%(file.replace('\\', '/'), label))
    else:
        for file in files:
           f.write('%s\n'%file.replace('\\', '/'))       
    f.close()

def get_labels(file_list):
    labels = []
    for file in file_list:
        label = os.path.basename(file).split('_')[0]
        labels.append(label)
        
    return labels
    
def main():
    train_list = load('./train.txt')
    y_train = get_labels(train_list)
    print(len(train_list)) 

    for index in range(len(train_list)):
        train_list[index] = train_list[index].split(' ')[0]
        
    save_as_listfile('train_file_list.txt', train_list)
    save_as_listfile('train_label.txt', y_train)
  
    
if __name__ == '__main__':
    main()