# SGNS-Pytorch
Skip-Gram and Negative Sampling for Chinese dataset in pytorch

### Dependencies

* python 3.6 +

* pytorch 1.0 +

* Logging

* Jieba

* Genism

### How to use

- Download Chinese Wiki dataset from [zhwiki_download](https://dumps.wikimedia.org/backup-index.html), and save it in `data/`

- Preprocess for dataset：
  - Change the .xml file to .txt file: `python XML2txt.py`
  - Transform Traditional Chinese into simplified Chinese: `python tans_t2s.py`
  - Chinese word segmentation: `python seg_wiki.py`  
  
- Train your model:
  - Train a model by using `gensim`: `python SGNS_train.py`
  - Train our SGNS model: `python train.py`