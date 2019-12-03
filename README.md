# Chainer RetinaNet

Chainer implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

There is other framework RetinaNet implementations ([Keras](https://github.com/fizyr/keras-retinanet), [PyTorch]()).


## Result

### MS COCO2017 Val

I have not yet evaluate in COCO because I do not have enough computational resources to train COCO in local environments.
I plan to use GCP to train models.
But I need some time because I am a student and do not afford to use it, so please wait.
Sorry...

If you train models in COCO using this repository, it would be very helpful if you send pre-trained model with config file used when training.

| Model | Train dataset | mmAP |
|:-:|:-:|:-:|
| RetinaNet ResNet50 | COCO2017 train | - |
| RetinaNet ResNet101 | COCO2017 train | - |

### PASCAL VOC2007 Test

| Model | Train dataset | mAP (PASCAL VOC2007 metric) |
|:-:|:-:|:-:|
| RetinaNet ResNet50 | VOC2007\&2012 trainval | 75.6 % |
| RetinaNet ResNet101 | VOC2007\&2012 trainval | 77.3 % |

## Pre-trained model

COCO pre-trained model is not yet available because I do not have enough computational resources to train COCO in local environments.
I plan to use GCP to train models.
But I need some time because I am a student and do not afford to use it, so please wait.
Sorry...

If you train models in COCO using this repository, it would be very helpful if you send pre-trained model with config file used when training.

| Train dataset | Model |
|:-:|:-:|
| COCO2017 train | [~~RetinaNet ResNet50~~]()
| COCO2017 train | [~~RetinaNet ResNet101~~]()
| VOC2007\&2012 trainval | [RetinaNet ResNet50](https://drive.google.com/open?id=1jQJSnkMidiIzQnulwK8VgOum3AGnEHDy) |
| VOC2007\&2012 trainval | [RetinaNet ResNet101](https://drive.google.com/open?id=1Bg3_8i3BIQcHoFHPoxGdC3ehZ215zmH-) |

## Installation

You can choose two options: Docker and Local.
I recommend to execute a train script in a docker container because ChainerMN saves a lot of training time, CUDA-Aware MPI (ChianerMN need CUDA-Aware MPI) installing is a little messy, and you have to install it as root user.

### Docker

Install docker and nvidia-docker2.
Please reference to [nvidia-docker2 installation guide](https://github.com/NVIDIA/nvidia-docker)

If you use docker, add `--allow-run-as-root` option when use mpiexec.

#### Pull already built image

There is docker images built by me.
Plase pull docker images.

```bash
docker pull <image_path>
```

Follwoing docker images are avairable.

| OS | CUDA Version | Image path |
|:-:|:-:|:-:|
| Ubuntu18.04 | CUDA10.1 | hymldocker/chainer:6.5.0-ubuntu18.04-cuda10.1 |
| Ubuntu16.04 | CUDA10.1 | hymldocker/chainer:6.5.0-ubuntu16.04-cuda10.1 |

#### Build image

You can build a docker image by yourself.

```bash
cd docker/<image/dir>
docker build . -t <image_name:tag>
```

### Local

Assume that you use Anaconda and python3.

1. Clone this repository.
2. (Optional) Install CUDA-Aware MPI when you use multi GPUs.
Please reference to [ChainerMN installation guide](https://chainermn.readthedocs.io/en/stable/installation/guide.html).
If you do not train model, you will not need CUDA-Aware MPI.
3. Fix requirements.txt.

    - Rewrite cuda-cuda100 of requirements.txt according to your CUDA version.
For example, rewrite to cupy-cuda92 if your CUDA version is CUDA9.2.
    - Remove mpi4py from requirements.txt if you do not install CUDA-Aware MPI.

4. Install required python packages.

```bash
pip install -r requirements.txt
```

## Preparation

Fix `coco_dir` and `voc_dir` in `configs/path_catalog.py` according to path to the datasets in your environments.
If you already downloaded the datasets, these are automatically loaded.
If you do not yet download the datasets, these are automatically downloaded to `coco_dir` and `voc_dir`.
Therefore, you do not need to donwload COCO and VOC.

## Demo

If you want to demo using models trained by me, please specify --pretrained_model as auto.
If spefied this option, models trained by me is donwloaded and used automatically.

Demo with evaluation dataset.

```bash
python demo.py <path/to/config> --pretrained_model <path/to/pretrained_model>
```

Demo with saved images.

```bash
python demo.py <path/to/config> --pretrained_model <path/to/pretrained_model> --indir <path/to/directory/images>
```

Demo with webcam.

```bash
python demo.py <path/to/config> --pretrained_model <path/to/pretrained_model> --webcam
```

## Evaluation

If you want to evaluate using models trained by me, please specify --pretrained_model as auto.
If spefied this option, models trained by me is donwloaded and used automatically.

Evaluate using a sigle GPU.

```bash
python evaluate.py <path/to/config>
```

Evaluate using multi GPUs.

```bash
mpiexec -n N_GPU python evaluate_multi.py <path/to/config>
```

## Train

Train using a sigle GPU.

```bash
python train.py <path/to/config>
```

Train using multi GPUs (recomended).

```bash
mpiexec -n N_GPU python train_multi.py <path/to/config>
```

When you execute evaluation and demo program with trained model yourself, the last iteration model is automatically loaded if you do not specified pretrained model path.


## Acknowledgements

- I refererenced to other framework RetinaNet implementations ([fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet), [yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet))
- Significant amounts of code are borrowed from the [chainer/chainercv](https://github.com/chainer/chainercv)