# Hyper Face #
[Hyper Face](https://arxiv.org/abs/1603.01249) implementation which predicts face/non-face, landmarks, pose and gender simultaneously.

This is NOT official implementation.

This software is released under the MIT License, see LICENSE.txt.

## Features ##
* `Chainer` implementation
* Image viewer on web browsers

## Testing Environments ##
### Ubuntu 16.04 ###
* Python 2.7
* Chainer 1.14.0
* OpenCV 2.4.9
* Flask 0.11.1
* Flask_SocketIO 2.4
* Dlib 19.1.0

### Arch Linux ###
* Python 3.5
* Chainer 1.14.0
* OpenCV 3.1.0
* Flask 0.10.1
* Flask_SocketIO 2.2
* Dlib 19.1.0

## Configuration ##
Important variables are configured by `config.json`.

Set `gpu` positive number to use GPU, port numbers of web servers and so on.

## Train ##
### Preparation ###
Download [AFLW Dataset](https://lrs.icg.tugraz.at/research/aflw/) and [AlexNet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet), expand them and set `aflw_sqlite_path`, `aflw_imgdir_path`, and `alexnet_caffemodel_path` in `config.json`

### Pre-training ###
Pre-training with RCNN_Face model.
```bash
python ./scripts/train.py --pretrain
```
Open `http://localhost:8888/`, `http://localhost:8889/` and `http://localhost:8890/` with your web browser to see loss graphs, network weights and predictions.
Port numbers are configured by `config.json`.

### Main training ###
```bash
python ./scripts/train.py --pretrainedmodel result_pretrain/model_epoch_40
```
Use arbitrary epoch number instead of 40.

## Test ##
To skip training, please use trained model from [here](https://drive.google.com/file/d/1w5MX5VRvGZCHfEo6nclDrgGOGqs_w7ag/view?usp=sharing) (___Do not expand___ as zip).

### AFLW test images ###
```bash
python ./scripts/use_on_test.py --model model_epoch_190
```
Open `http://localhost:8891/` to see predictions.
<img src="https://raw.githubusercontent.com/takiyu/hyperface/master/sample_images/face.png">
<img src="https://raw.githubusercontent.com/takiyu/hyperface/master/sample_images/nonface.png">

### Your image file ###
Set your image file with `--img` argument.
The dependence are less than other tests and demos.
```bash
python ./scripts/use_on_file.py --model model_epoch_190 --img sample_images/lena_face.png
```
Input images are contained in `sample_images` directory.

<img src="https://raw.githubusercontent.com/takiyu/hyperface/master/sample_images/lena_face_result.png">
<img src="https://raw.githubusercontent.com/takiyu/hyperface/master/sample_images/lena_face_result2.png">

## Demos with post-processes ##
Open `http://localhost:8891/` to see demos.

### AFLW test images ###
```bash
python ./scripts/demo_on_test.py --model model_epoch_190
```
Demo using AFLW test images
<img src="https://raw.githubusercontent.com/takiyu/hyperface/master/sample_images/demo1.png">

### Web camera on your browser ###
```bash
python ./scripts/demo_live.py --model model_epoch_190
```

## ToDo ##
- [ ] Tune training parameters.
- [ ] Fix pose drawing.
- [x] Implement post processes.
- [ ] Tune post processes parameters.
