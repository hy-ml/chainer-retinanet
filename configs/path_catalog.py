import os


# change these variables into the path to the dataset in your environment
coco_dir = os.path.expanduser('~/dataset/coco2017')
voc_dir = os.path.expanduser('~/dataset/voc')

outdir = './out'
logdir = './log'

gdrive_ids = {
    'VOC': {
        'RetinaNetResNet50': '1jQJSnkMidiIzQnulwK8VgOum3AGnEHDy',
        'RetinaNetResNet101': '1Bg3_8i3BIQcHoFHPoxGdC3ehZ215zmH-',
    },
    'COCO': {
        'RetinaNetResNet50': '',
        'RetinaNetResNet101': '',
    },
}
