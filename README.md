# Avantgarde Challange Results
|                      | KAGGLE DATASET (MAE) | FACEBOOK DATASET (MAE) | KAGGLE DATASET (RMSE) | FACEBOOK DATASET (RMSE) |
|----------------------|:--------------------:|:----------------------:|-----------------------|-------------------------|
|   CORAL-CNN (AFAD)   |        12.446        |         15.075         |         15.073        |          19.393         |
|   CORAL-CNN (CACD)   |        12.512        |         16.222         |         15.151        |          18.997         |
|  CORAL-CNN (MORPH)   |         9.509        |         11.478         |         11.637        |          14.128         |
| DLDL-v2 (TinyAgeNet) |         7.365        |          8.785         |         9.280         |          11.136         |
|    DOLD (UTKFace)    |         8.385        |         12.506         |         10.387        |          10.631         |

# Preprocess of Datasets
### Kaggle:
----------
- All of the images of the age prediction dataset on Kaggle are cropped and aligned. No need to preprocess. Test data has 7008 cropped and aligned face images.
### Facebook:
-------------
- Facebook dataset contains high quality videos of diverse set of age, genders, apparent skin tones and ambient lighting conditions.

- About 100GB video data have been preprocessed for the challenge. First, frames are extracted from videos. Roughly, 20 frames have been extracted from a video. 6610 images are extracted. These extracted images are not croppped and aligned. These need to be cropped and aligned in order to evaluate models. For this purpose, I used FACENET {[Code](https://github.com/davidsandberg/facenet/), [Paper](https://arxiv.org/abs/1503.03832)} to crop and align images. I used this [shape predictor](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and [instruction](https://github.com/AaltoVision/img-transformer-chain/blob/master/README.md#face-alignment--cropping) in order to run FACENET.
# Models and Evaluation
- CORAL-CNN {[Code](https://github.com/Raschka-research-group/coral-cnn), [Paper](https://arxiv.org/abs/1901.07884)}: The model is trained with [MORPH](https://paperswithcode.com/dataset/morph), [CACD](https://paperswithcode.com/dataset/cacd) and [AFAD](https://paperswithcode.com/dataset/afad) datasets. As part of the challange, all pre-trained models of CORAL-CNN are evaluated on [Facebook](https://ai.facebook.com/datasets/casual-conversations-dataset) and [Kaggle](https://www.kaggle.com/mariafrenti/age-prediction) datasets.

- DLDL-v2 {[Code](https://github.com/gaobb/DLDL-v2), [Paper](https://www.ijcai.org/proceedings/2018/0099.pdf)}: The model is trained with [ChaLearn 2015](https://paperswithcode.com/sota/age-estimation-on-chalearn-2015) dataset. I got errors during the evaluation phase of DLDL-v2-Torch model. Train and evaluation codes are written in Lua and the repo is not maintained. I solved all of the issues but one. I have opened [issue](https://github.com/gaobb/DLDL-v2/issues/3) for it and there is no reply till now. Rather than getting stuck at this point, I found a [PyTorch](https://github.com/PuNeal/DLDL-v2-PyTorch) implementation of DLDL-v2 model. The pretrained [TinyAgeNet](https://github.com/PuNeal/DLDL-v2-PyTorch/blob/master/pretrained/TinyAge_chalearn.pt) is used for evaluation.

- DOLD {[Code](https://github.com/axeber01/dold), [Paper](https://arxiv.org/abs/2006.15864)}: The model is trained with [UTKFace](https://paperswithcode.com/dataset/utkface) dataset. There is no pretrained weigths for the model therefore I  trained dold model on UTKFace. It is evaulated on Kaggle and Facebook datasets.

# Best Model
- According to [results](#avantgarde-challange-results):
- Best model is [DLDL-v2 PyTorch](https://github.com/PuNeal/DLDL-v2-PyTorch).
