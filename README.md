# cbow-pytorch 

### Introduction
In this repo, I implement a CBOW (Continuous Bag of Words) model based on the [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) paper using Pytorch.

### Folder hierarchy
```
.
├── config
│   └── config_file.yaml
├── data
│   └── raw_data.txt
├── model
│   └── cbow.py
├── README.md
├── train.py
└── utils
    └── hparams.py

4 directories, 6 files
```

### Training
```
$ git clone https://github.com/HiepThanh0510/cbow-pytorch

$ python3 train.py --config config/config_file.yaml
```

### Example: 
Corpus:
```
Manga Moushoku Tensei, also known as "Jobless Reincarnation," is a Japanese 
manga series that has captivated readers with its compelling storytelling and 
captivating artwork. Originally written as a light novel by Rifujin na Magonote, 
the manga adaptation was illustrated by Yuka Fujikawa and published by Media Factory. 
This fantasy isekai manga follows the journey of Rudeus Greyrat, a 34-year-old 
unemployed and socially awkward man who meets an unfortunate end, only to be 
reincarnated into a new world as a baby named Rudeus. Determined to make the 
most of his second chance at life, Rudeus retains the knowledge of his past 
life and strives to grow stronger, both mentally and physically, as he embarks 
on an epic adventure filled with magic, monsters, and self-discovery.
```
Context: 
```
This, fantasy, manga, follows
```
Output:
```
isekai
```