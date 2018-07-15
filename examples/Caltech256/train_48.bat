@echo off 

set TOOLS=D:/Tools/Caffe/bin

%TOOLS%/caffe train -solver  solver_CALTECH256_48.prototxt -weights ../../../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0