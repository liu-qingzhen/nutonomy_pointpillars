 修改过后的pointpillars
 
 @[TOC](Pointpillars 配置教程)

# Pointpillars 介绍
源码地址：https://github.com/nutonomy/second.pytorch
Pointpillars是nutonomy基于Second改进的3d目标检测算法


## 配置
ubuntu 18.04 + RTX2080ti + CUDA 10.1 + cuDNN 7.5 + anaconda 3 + pytorch 1.5 

### 下载pointpillar
```
git clone https://github.com/nutonomy/second.pytorch.git
```
之所以文件叫second是因为pointpillars是基于second做的改进

这个clone的文件会有一些问题 里面缺少了c++编译后的文件
直接跑会报错缺少nms文件和box_ops_cc文件，如果使用直接clone的nutonomy的pointpillars
这个问题卡了我好久，报错要从头看
可以到我的github把完整的clone下来
```
git clone https://github.com/liu-qingzhen/nutonomy_pointpillars
```
因为是用python3.6编译的 所以conda创建环境是需要设置python=3.6

也可以到github上到这个repo里根据报错找到缺失的文件自行补全
```
https://github.com/SmallMunich/nutonomy_pointpillars
```
### 安装 dependencies
```
conda create -n pointpillars python=3.6 anaconda
source activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
pip install torch==1.5 (我最终程序的torch版本是1.5，安装spconv的时候如果安装不成功可是试一试不同版本，我试了1.1和1.5）
pip install fire tensorboardX
conda install google-sparsehash -c bioconda
```
### 安装 spconv
> This is not required for PointPillars, but the general SECOND code base expects this to be correctly configured.
>
spconv不是pointpillars必须的 但是second需要 所以还是需要安装 这个会比较麻烦 需要耐心一点

首先安装spconv需要Cmake版本高于3.12.2 我最后就用了3.12.2 没有测试过3.19可不可以

下载cmake 具体版本自己调
```
wget https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.tar.gz
```
之后的操作可以参考网上cmake安装的教程 最好一遍安装成功 不大容易删干净
安装好看一下cmake的版本
```
cmake --version
```


#### clone sponcv到本地
```
git clone https://github.com/traveller59/spconv.git --recursive
```
#### 安装 Boost Geometry
```
sudo apt-get install libboost-all-dev
```
~~~
cd spconv
python setup.py bdist_wheel
cd dist
pip install 文件夹里的spconv文件
~~~
可能会报错
```
No CMAKE_CUDA_COMPILER could be found
```
终端输入
```
export PATH=/usr/local/cuda/bin:$PATH
```
如果还是不能安装完成可以试试改变torch版本
如果还有问题，可以github搜一下spconv 1.0安装 我用的最新版1.2.1没问题

如果用torch 1.5 报错驱动比较老
打开 software update manager 左下角 settings
选择 additional Drivers 
更新驱动 重启电脑


### setup cuda for numda
```
gedit ~/.bashrc
添加到最后
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
保存关闭
source ~/.bashrc
```
### 添加second到python路径
不添加路径的话无法从文件里import

```
gedit ~/.bashrc
export PYTHONPATH=$PYTHONPATH:${存放second的路径}$/second/second.pytorch
source ~/.bashrc
```

### 下载配置kitti数据集
可以到kitti官网下载 不过好像需要提交申请
可以在终端输入以下命令
```
# download left color images:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

# download calibration results:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

# download labels:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# download Velodyne point clouds:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

# download development kit:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip
```
可以放在second文件夹里 也可以放在其他地方 反正最后都需要改配置文件里的路径

数据集的存放格式按照nutonomy给出的方式
```
└── KITTI_DATASET_ROOT <-- 这个是你的文件夹名称 可以改成别的名字
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
  ```

2. Create kitti infos:
**这里的--data_path需要自己改成你的数据集存放文件夹的绝对路径，也就是上面的文件夹结构的KITTI_DATASET_ROOT**

```
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

4. Create reduced point cloud:

```
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```
5. Create groundtruth-database infos:

```
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```
创建这些文件应该需要一些时间 耐心等待 遇到报错按照错误来调整就行 不会有大问题

6. 更改配置文件
这里nutonomy的readme写的也不是很详细，配置文件里的所有路径都需要更改，不止是他提到的这几个
比如之后训练需要用到./configs/pointpillars/car/xyres_16.proto这个文件
就需要把里面所有的路径都改成你的绝对路径
```
train_input_reader: {
  record_file_path: "/media/lzy/T7/KITTI_DATASET_ROOT/kitti_train.tfrecord"
  ...
  ```
  比如我用的移动硬盘来存放数据集 就改成这样 其他路径类似
 
 ## 训练
 打开终端
 `
 cd ~/second.pytorch/second
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=./model_dir
`
如果要训练新的模型 需要保证--model_dir这个路径之前没有训练过 不然会在这个基础上训练

### 训练的时候可能会遇到的问题
1. 需要bool型 代码是一个byte

``second.pytorch\second\pytorch\models``文件夹下的``voxelnet.py``的911行

改为：
`` opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()``

2. TypeError: 'numpy.float64' object cannot be interpreted as an integer

给numpy版本降级
``pip install -U numpy==1.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple``

## 总结
大概就是这么多，我比较菜，在spconv和cmake上卡了两天，最后终于配置好
目前还在训练，之后可能会在Nvidia Drive Xavier上做一下inference，后续再更新
有什么其他的问题可以私信vx讨论。

## 参考
 ```
 [1]: https://cloud.tencent.com/developer/article/1629517
 [2]: https://blog.csdn.net/qimingxia/article/details/103097986
 [3]: https://blog.csdn.net/qq_40092110/article/details/105242996
 [4]: https://github.com/nutonomy/second.pytorch
 ```
