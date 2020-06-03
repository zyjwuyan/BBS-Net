# BBS-Net
BBS-Net: RGB-D Salient Object Detection with
a Bifurcated Backbone Strategy Network

## Requirements

Python 3.7, Pytorch 0.4.0+, Cuda 10.0, TensorboardX 2.0, opencv-python

## Data Preparation

 - Download the raw data from [Here](https://pan.baidu.com/s/1SxBjlTF4Tb74WjuDsRmM3w) [code: yiy1] and trained model (BBSNet.pth) from [Here](https://pan.baidu.com/s/1Fn-Hvdou4DDWcgeTtx081g) [code: dwcp]. Then put them under the following directory:
 
        -BBS_dataset\ 
          -RGBD_for_train\  
          -RGBD_for_test\
          -test_in_train\
        -BBSNet
          -models\
          -model_pths\
             -BBSNet.pth
          ...
            
## Training & Testing

Train the BBSNet:

    `python BBSNet_train.py --batchsize 10 --gpu_id 0 `

Test the BBSNet:

    `python BBSNet_test.py --batchsize 10 --gpu_id 0 `