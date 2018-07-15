"""
--------------------------------------------------------
Deep Learning of Binary Hash Codes for Fast Image Retrieval on Windows
Another link:
    SemanticRetrieval:
    
Written by xiaowangdu [https://github.com/xiaowangdu]
--------------------------------------------------------
"""

from __future__ import print_function
import sys, os
import caffe
import time
import numpy as np
from prepare_batch import prepare_batch
import scipy.io as scio
import file2list
import math

def pycaffe_batch_feat(im_name_list, use_gpu, feat_len, model_def_file, model_file, root_dir):
    """
    Extract image feature by batchs.
    
    Using net.forward().

    :Args:
     - im_name_list - list of image files.
     - use_gpu - choose cpu or gpu.
     - feat_len - length of image features.
     - model_def_file - the deploy model file.
     - model_file - the caffemodel file.
     - root_dir - root directory.
    """
    batch_size = 10
    dim = feat_len
    num_im = len(im_name_list)

    if num_im % batch_size != 0:
        print('Assuming batches of {:d} images, rest({:d}) will be filled with zeros'.format(batch_size, num_im % batch_size))

    # init caffe network (spews logging info)
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    net = caffe.Net(model_def_file, model_file, caffe.TEST)
    
    # load mean file
    mean_file_pth = os.path.join(root_dir, 'ilsvrc_2012_mean.mat')
    image_mean = scio.loadmat(mean_file_pth)['image_mean'].astype(float)
    
    # prepare input
    scores = np.zeros((num_im, dim), dtype=float)
    num_batches = math.ceil(num_im/batch_size) # should be the same as ceiling
    initic=time.time()
    for batch in range(num_batches):
        start = batch * batch_size
        length = min(num_im - start, batch_size)
        end = start + length

        '''
        rg = np.arange(batch_size*batch, min(num_im, batch_size*(batch+1)))  # range
        batch_im_name_list = []
        for i in rg:
            im_name = im_name_list[i]
            batch_im_name_list.append(im_name)
        '''
        
        input_data = prepare_batch(im_name_list[start:end], image_mean, batch_size)

        t_consume = time.time() - initic
        print('Batch {:d} out of {:d} {:.2f}% Complete ETA {:.2f} seconds'.format( \
            batch, num_batches, (batch + 1) * 1.0 / num_batches * 10, t_consume / (batch + 1) * (num_batches-batch) \
        ))
        net.blobs['data'].data[...] = input_data
        output_data = net.forward()

        raw_score = output_data['fc8_kevin_encode']
        ok_score = np.round(raw_score.astype(float), 4)
        
        if length < batch_size:
            scores[start:end, :] = ok_score[0:length, :]
            break
            
        scores[start:end, :] = ok_score

    return scores, im_name_list


if __name__ == '__main__':
    """
    TEST
    """
    
    root_dir = 'E:/GitHub/cvprw15-win/'  #e.g 
    root_dataset = 'E:/GitHub/dataset/cifar10/'
    
    feat_len = 48  
    use_gpu = 1
    # models
    model_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/KevinNet_CIFAR10_48.caffemodel')
    model_def_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/KevinNet_CIFAR10_48_deploy.prototxt')

    # img_list
    test_file_list = os.path.join(root_dir, 'py/img_list.txt')

    scores, im_list = pycaffe_batch_feat(file2list.load(test_file_list, root_dataset), use_gpu, feat_len, model_def_file, model_file, root_dir)
    scores = (scores > 0.5).astype(int)
    scio.savemat('dataNew.mat', {'binary_codes': scores.T, 'list_im': np.array(im_list)}) 
    print(scores[10, :])