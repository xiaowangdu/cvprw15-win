@echo off 

set EXAMPLE=E:/GitHub/cvprw15-win/examples/cvprw15-cifar10
set DATA=E:/GitHub/dataset/cifar10/examples/cvprw15-cifar10/dataset/
set TOOLS=D:/Tools/Caffe/bin

set TRAIN_DATA_ROOT=E:/GitHub/dataset/cifar10/examples/cvprw15-cifar10/dataset/
set VAL_DATA_ROOT=E:/GitHub/dataset/cifar10/examples/cvprw15-cifar10/dataset/

set RESIZE=true

set RESIZE_HEIGHT=0
set RESIZE_WIDTH=0

if "%RESIZE%"=="true" (
	set RESIZE_HEIGHT=256
	set RESIZE_WIDTH=256
)

if not exist %TRAIN_DATA_ROOT% (
	echo Error: TRAIN_DATA_ROOT is not a path to a directory: %TRAIN_DATA_ROOT%
	echo Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path where the ImageNet training data is stored.
	pause
	exit -1
)

if not exist %VAL_DATA_ROOT% (
	echo Error: VAL_DATA_ROOT is not a path to a directory: %VAL_DATA_ROOT%
	echo Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path where the ImageNet training data is stored.
	pause
	exit -1
)

echo Creating train leveldb...

%TOOLS%/convert_imageset --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --backend="leveldb" %TRAIN_DATA_ROOT% %DATA%/train.txt %EXAMPLE%/cifar10_train_leveldb

echo Creating val leveldb...

::%TOOLS%/convert_imageset --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --backend="leveldb" %VAL_DATA_ROOT% %DATA%/val.txt %EXAMPLE%/cifar10_val_leveldb

echo Done.

pause