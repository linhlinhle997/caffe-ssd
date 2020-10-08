# caffe-ssd
`GeForce GTX 1070 - NVIDIA Driver 440 - Cuda 10.2`
## Installation
1. Install dependencies
- OpenCV
``` Shell
  $ sudo apt install python3-opencv
```
- ATLAS or BLAS
``` Shell
# Atlas
$ sudo apt-get install libatlas-base-dev 
```
``` Shell
# OpenBLAS
$ sudo apt-get install libopenblas-dev 
```
- Other dependecies
``` Shell
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install — no-install-recommends libboost-all-dev
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
$ sudo pip3 install protobuf
$ sudo apt-get install the python3-dev
```
1. Get the code of [caffe](https://github.com/linhlinhle997/caffe-ssd.git)
``` Shell
  $ git clone https://github.com/linhlinhle997/caffe-ssd.git
  $ cd caffe-ssd
```
2. Change `Makefile.config`
``` Shell$ 
  $ cd ./caffe-ssd
  $ cp Makefile.config.example Makefile.config
  $ nano Makefile.config
```
- Change `Makefile.config` for Cuda-10.2
``` Shell
  5. #USE_CUDNN := 1
  => USE_CUDNN := 1

  11. #USE_OPENCV:=0 
  => USE_OPENCV := 1

  21. #OPENCV_VERSION := 3
  => OPENCV_VERSION := 3

  25. #CUSTOM_CXX := g++ 
  => CUSTOM_CXX := g++
  
  36. CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
             -gencode arch=compute_20,code=sm_21 \
             -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_61,code=sm_61
  => CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_37,code=sm_37 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_61,code=sm_61 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_75,code=compute_75

  50. #BLAS := atlas
      BLAS := open
  => BLAS := atlas
    #BLAS := open

  69. PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include
  => #PYTHON_INCLUDE := /usr/include/python2.7 \
	#/usr/lib/python2.7/dist-packages/numpy/core/include
  
  79. # PYTHON_LIBRARIES := boost_python3 python3.5m
    # PYTHON_INCLUDE := /usr/include/python3.5m \
    #/usr/lib/python3.5/dist-packages/numpy/core/include
  => PYTHON_LIBRARIES := boost_python3 python3.6m
    PYTHON_INCLUDE := /usr/include/python3.6m \ 
    /usr/lib/python3.6/dist-packages/numpy/core/include

  95. INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
      LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
  => INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
      LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```
3. Install Python Caffe dependencies
``` Shell
$ cd ./python
$ for req in $(cat requirements.txt); do pip3 install $req; done
$ cd ..
```
- Add the module directory to `~/.zshrc` with `export PYTHONPATH=~/caffe-ssd/python:$PYTHONPATH`, run command `source ~/.zshrc`
4. Build the code.
``` Shell
    $ make -j8
    $ make py
    $ make test -j8
    $ make runtest -j8
```
5. Downloads KITTI Dataset `data_object_image_2`, `data_object_label_2`. Modify KITTI to PASCAL VOC file format
- Place in `~/caffe-ssd/data/KITTIdevkit/KITTI/`. Copy images in `data_object_image_2/training/image_2` to `JPEGImages`, copy labels in `data_object_label_2/training/label_2` to `Labels`. Final run 3 file python, we have folder `.xml` and split data to `trainval` set, `test` set.
``` Shell
    $ cd ./data/KITTIdevkit/KITTI
    $ python3 modify_annotations_txt.py 
    $ python3 txt_to_xml.py
    $ python3 create_train_test_txt.py
```
- Place in `~/caffe-ssd/`. After run the scripts in folder `./data/KITTI`, will generate two `LMDB` file paths `~/caffe-ssd/examples/KITTI/KITTI_test_lmdb` and `~/caffe-ssd/examples/KITTI/KITTI_trainval_lmdb`
``` Shell
    $ cd ~/caffe-ssd
    $ chmod +x ./data/KITTI/create_list_kitti.sh
    $ ./data/KITTI/create_list_kitti.sh
    $ chmod +x ./data/KITTI/create_data_kitti.sh
    $ ./data/KITTI/create_data_kitti.sh 
```
- Work tree
``` Shell
~/caffe-ssd/data/
    ├── KITTIdevkit/
    │   └── KITTI/
    │       ├── Annotations/
    │       │   ├── 000000.xml
    │       │   └── ...
    │       ├── ImageSets/
    │       │       ├── trainval.txt
    │       │       ├── test.txt
    │       │       └── ...
    │       ├── JPEGImages/
    │       │   ├── 000000.png
    │       │   └── ...
    │       ├── Labels/
    │       │   ├── 000000.txt
    │       │   └── ...
    │       ├── lmdb/
    │       │   ├── KITTI_test_lmdb/
    │       │   │   ├── data.mdb
    │       │   │   └── lock.mdb
    │       │   └── KITTI_trainval_lmdb/
    │       │       ├── data.mdb
    │       │       └── lock.mdb
    │       ├── create_train_test_txt.py
    │       ├── modify_annotations_txt.py
    │       └── txt_to_xml.py
    └── KITTI/
        ├── create_data_kitti.sh
        ├── create_list_kitti.sh
        ├── labelmap_kitti.prototxt
        ├── test.txt
        ├── test_name_size.txt
        └── trainval.txt
```
6. Downloads VGG16 pre-train model from [VGG16](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6) and place in folder `~/caffe-ssd/models/VGGNet/`
7. Training model
``` Shell
    $ cd  ~/caffe-ssd
    $ python3 ./examples/ssd/ssd_pascal_kitti.py
```
8. Testing model
``` Shell
    $ python3 ./examples/ssd/ssd_detect_kitti.py
```
## Reference
- Caffe framework: [https://caffe.berkeleyvision.org/installation.html](https://caffe.berkeleyvision.org/installation.html)
- Caffe ssd from weiliu89: [https://github.com/weiliu89/caffe](https://github.com/weiliu89/caffe)
- Toturial SSD detector training KITTI data set of Jesse_Mx: 
    - [https://blog.csdn.net/Jesse_Mx/article/details/65634482](https://blog.csdn.net/Jesse_Mx/article/details/65634482) 
    - [https://blog.csdn.net/Jesse_Mx/article/details/70048255](https://blog.csdn.net/Jesse_Mx/article/details/70048255)