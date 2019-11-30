# Chainer RetinaNet

Chainer implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

There is other framework RetinaNet implementations ([Keras](https://github.com/fizyr/keras-retinanet), [PyTorch]()).

## Result

### MS COCO2017 Val

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

| Train dataset | Model |
|:-:|:-:|
| COCO2017 train | [RetinaNet ResNet50](https://drive.google.com/open?id=1jQJSnkMidiIzQnulwK8VgOum3AGnEHDy)
| COCO2017 train | [RetinaNet ResNet101](https://drive.google.com/open?id=1Bg3_8i3BIQcHoFHPoxGdC3ehZ215zmH-)
| VOC2007\&2012 trainval | [RetinaNet ResNet50]() |
| VOC2007\&2012 trainval | [RetinaNet ResNet101]() |

## Installation

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

## Demo

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