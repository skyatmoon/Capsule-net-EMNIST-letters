# CapsNet-Pytorch-EMNIST-letters

A Pytorch implementation of CapsNet in the paper:   
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   

Thanks for https://github.com/XifengGuo/CapsNet-Pytorch
 
**TODO**
- Conduct experiments on other datasets. 
- Explore interesting characteristics of CapsuleNet.
- Test the reconstruction results on the EMNIST-letters

## Usage: Same as the https://github.com/XifengGuo/CapsNet-Pytorch

**Step 1. Train a CapsNet on MNIST**  

Training with default settings:
```
python capsulenet.py
```

Launching the following command for detailed usage:
```
python capsulenet.py -h
``` 

**Step 2. Test model and show reconstruction results**

Suppose you have trained a model using the above command, then the trained model will be
saved to `result/trained_model.pkl`. Now just launch the following command to get test results.
```
python capsulenet.py --testing --weights result/trained_model.pkl
```
It will output the testing accuracy and show the reconstructed images

## Results       

**Reconstruction result**  

Digits at top 5 rows are real images from EMNIST and 
digits at bottom are corresponding reconstructed images.

All the results are based on 5 epochs traing.
Time for training is 480s/epoch on  GTX1060

Results can showing by both one-channel and color

![](result/real_and_recon.png)![](result/real_and_recon_color.png)
