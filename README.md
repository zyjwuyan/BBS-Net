# BBS-Net
BBS-Net: RGB-D Salient Object Detection with
a Bifurcated Backbone Strategy Network

## 1. Requirements

Python 3.7, Pytorch 0.4.0+, Cuda 10.0, TensorboardX 2.0, opencv-python

## 2. Data Preparation

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
            
## 3. Training & Testing

Train the BBSNet:

    `python BBSNet_train.py --batchsize 10 --gpu_id 0 `

Test the BBSNet:

    `python BBSNet_test.py --gpu_id 0 `
## 4. Results
### 4.1 Qualitative Comparison
<p align="center">
    <img src="Images/resultmap.png"/> <br />
 <em> 
    Figure 2: Qualitative visual comparison of the proposed model versus 8 SOTA
models.
    </em>
</p>
<p align="center">
    <img src="./Images/detailed-comparison.png"/> <br />
 <em>
  Table 1: Quantitative comparison of models using S-measure ($S_/alpha$), max F-measure
(Fβ), max E-measure (Eξ) and MAE (M) scores on 7 datasets. 
  </em>
</p>
<!--
|  Dataset  | NJU2K  | NLPR | STERE |DES    |LFSD  |SSD |SIP|
|  -------  | -----  |----  |-----  |---    |----  |---  |---|
| S-measure |.921    |.930  |.908   |.933  | .864  | .882|.879 |
| F-measure |.920    |.918  |.903   |.927  | .859  | .859|.883 |
| E-measure |.949    |.961  |.942   |.966  | .901  | .919|.922 |
| MAE       | .035   |.023  |.041   |.021  | .072  | .044|.055 |
-->
### 4.2 Download
Test map of the above datasets can be download from [here](https://pan.baidu.com/s/1O-AhThLWEDVgQiPhX3QVYw) [code: qgai ]
## 5. Citation

Please cite the following paper if you use this repository in your reseach.

## 6. Acknowledgement
We implement this project based on the code of ‘Cascaded Partial Decoder for Fast and Accurate Salient Object Detection, CVPR2019’ proposed by Wu et al.
