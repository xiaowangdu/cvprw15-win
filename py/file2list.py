"""
--------------------------------------------------------
Deep Learning of Binary Hash Codes for Fast Image Retrieval on Windows
Another link:
    SemanticRetrieval:
    
Written by xiaowangdu [https://github.com/xiaowangdu]
--------------------------------------------------------
"""

import sys, os

def load(file_list_path, root_dir = ''):
    """
    load file list.
    """
    fin = open(file_list_path)
    images_list = []
    for line in fin.readlines():
        t = line.rstrip()
        if t[0]=='/':
            t = t[1:]
        im_name = os.path.join(root_dir, t)
        images_list.append(im_name)
    fin.close()
    return images_list