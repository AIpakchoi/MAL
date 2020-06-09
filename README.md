# MAL
MAL code for the paper Multiple Anchor Learning for Visual Object Detection [pdf](https://arxiv.org/abs/1912.02252).


## install
Get into MAL root folder.
1. Create conda env by `conda env create -n MAL` and activate it by 'conda activate MAL'.
2. Install python libraries.
`conda install ipython ninja yacs cython matplotlib tqdm`
3. Install pytorch 1.1 + torchvision 0.2.1 by pip.
download `whl` file at https://download.pytorch.org/whl/cu90/torch_stable.html
`pip install [downloaded file]`
4. Install pycocotools
`pip install pycocotools`
5. Copy https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark to this repository.
6. Build maskrcnn_benchmark by run `python setup.py build develop`
7. Install OpenCV3.

## inference for an image
1. Go to `./demo`
2. Run `python image_demo.py`. You can use your own image and change the image path in image_demo.py

## test on COCO dataset
Get into MAL root folder.
For test-dev set, run
`python python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file ./config/MAL_X-101-FPN_e2e.yaml MODEL.WEIGHT ./output/models/model_0180000.pth DATASETS.TEST "('coco_test-dev',)"`

For val set, run
`python python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file ./config/MAL_X-101-FPN_e2e.yaml MODEL.WEIGHT ./output/models/model_0180000.pth`

## experimental result
mAP = 47.0 on test-dev

## pre-trained model
ResNet50: https://share.weiyun.com/5kcZju5
ResNet101: https://share.weiyun.com/5gtr6Ho
ResNext101: https://share.weiyun.com/oUZUWfSx
