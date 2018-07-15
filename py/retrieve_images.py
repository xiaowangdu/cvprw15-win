"""
--------------------------------------------------------
Deep Learning of Binary Hash Codes for Fast Image Retrieval on Windows
Another link:
    SemanticRetrieval:
    
Written by xiaowangdu [https://github.com/xiaowangdu]
--------------------------------------------------------
"""

import sys, os
import caffe
import time
import numpy as np
import file2list
from precision import pdist2
from pycaffe_batch_feat import pycaffe_batch_feat
import scipy.io as scio
import math

class Config(object):
    """
    Save image retrieval configure.
    
    Use default config by Config().
    """
    def __init__(self):
        self.use_gpu = 1
        self.feat_len = 48  # number of activation output on hidden layer
        self.root_dir = 'E:/GitHub/cvprw15-win/' 
        self.dataset_dir = 'E:/GitHub/dataset/caltech256/'
        
        # models
        self.model_file = os.path.join(self.root_dir, 'examples/Caltech256/model_all_in_train_v2.0/KevinNet_CALTECH256_48_iter_50000.caffemodel')
        self.model_def_file = os.path.join(self.root_dir, 'examples/Caltech256/KevinNet_CALTECH256_48_deploy.prototxt')

        # images library
        self.img_train_list = os.path.join(self.root_dir, 'examples/Caltech256/model_all_in_train_v2.0/train-file-list.txt')
        self.binary_file = '{:s}/binary-train.mat'.format(os.path.join(self.root_dir, 'analysis/caltech256'))
        self.feat_train_file = '{:s}/feat-train.mat'.format(os.path.join(self.root_dir, 'analysis/caltech256'))
          
    def init_config(self, root_dir, dataset_dir,
                    model_file, model_def_file,
                    img_train_list, 
                    binary_file = None, feat_train_file = None
                    ):
        """
        Init retrieval config.
        """
        self.use_gpu = 1
        self.feat_len = 48
        self.root_dir = root_dir 
        self.dataset_dir = dataset_dir
        self.model_file = model_file
        self.model_def_file = model_def_file
        self.img_train_list = img_train_list
        if binary_file is not None:
            self.binary_file = binary_file
            
        if feat_train_file is not None:    
            self.feat_train_file = feat_train_file
     
     
configs = Config()

def retrieve_config(root_dir, dataset_dir,
                    model_file, model_def_file,
                    img_train_list, 
                    binary_file, feat_train_file):
    """
    Retrieval config.
    
    Usage:
        SemanticRetrieval
            -RetrievalTest.vcxproj
        
        Link: 
    """
    configs.init_config(root_dir = root_dir, dataset_dir = dataset_dir,
                        model_file = model_file, model_def_file = model_def_file, 
                        img_train_list = img_train_list, binary_file = binary_file, feat_train_file = feat_train_file)
    
  
def retrieve_images(target_image_path, top_k = 10):
    """
    Retrieval images.
    
    :Args:
     - target_image_path - target image.
     - top_k - the top k images retrieved.
    
    Usage:
        SemanticRetrieval
            -RetrievalTest.vcxproj
        
        Link: 
    """
    res = []

    img_lib = file2list.load(configs.img_train_list, configs.dataset_dir)
    
    if os.path.exists(configs.binary_file): # load .mat file
        binary_train = scio.loadmat(configs.binary_file)['binary_train'].astype(float)
        binary_train = binary_train.T
    else:
        feat_train, _ = pycaffe_batch_feat(img_lib, configs.use_gpu, configs.feat_len, configs.model_def_file, configs.model_file, configs.root_dir)
        scio.savemat(configs.feat_train_file, {'feat_train':feat_train.T})
        binary_train = (feat_train > 0.5).astype(int)
        scio.savemat(configs.binary_file, {'binary_train': binary_train.T})
    
    binary_target_img, _ = pycaffe_batch_feat([target_image_path], configs.use_gpu, configs.feat_len, configs.model_def_file, configs.model_file, configs.root_dir)
    binary_target_img = (binary_target_img > 0.5).astype(int)
    
    t_start = time.time()
    similarity = pdist2(binary_train, binary_target_img, 'hamming')
    t_consume = time.time() - t_start
    
    y2=np.argsort(similarity, kind = 'mergesort')
    for i in range(top_k):
        retrieval_path = img_lib[y2[i]]
        res.append(retrieval_path)
     
    #print(y2[0:top_k]) 
    return res   
    
def main():
    """
    TEST
    """
    target_image_path = 'F:/workspace/project/data/105_0028.jpg'
    res = retrieve_images(target_image_path)
    print(res) 
    
    
if __name__ == '__main__':
    main()