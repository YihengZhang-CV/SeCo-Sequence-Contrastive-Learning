# Sequence-Contrastive-Learning
This is the implementation of '**SeCo: Exploring Sequence Supervision for Unsupervised Representation Learning**' [AAAI 2021]. The original paper can be found at https://arxiv.org/abs/2008.00975 .

## Requirements
* torch
* torchvision
* liblinear

### Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a [MoCo](https://github.com/facebookresearch/moco) initialized ResNet-50 model, download the weights [MoCo v2 (200epochs)](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar) to the **pretrain** folder, and run:
```
bash main_train.sh
```

### Evaluation of linear classification
With a pre-trained model, to train a supervised linear SVM classifier on frozen features/weights, put the python interface of [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) into the **liblinear** folder, and run:
```
bash main_val.sh
```

## Citation
If you find this code useful for your research, please cite our paper:

    @inproceedings{yao2021seco,
      title={SeCo: Exploring Sequence Supervision for Unsupervised Representation Learning},
      author={Yao, Ting and Zhang, Yiheng and Qiu, Zhaofan and Pan, Yingwei and Mei, Tao},
      booktitle={35th AAAI Conference on Artificial Intelligence},
      year={2021}
    }
