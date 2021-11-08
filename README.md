# PMG-V2
 
Code release for "Progressive Learning of Category-Consistent Multi-Granularity Features for Fine-Grained Visual Classification" (IEEE T-PAMI 2021).

It is the extended version of "Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches" (ECCV2020): https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
 
### Requirement
 
python >= 3.6

PyTorch >= 1.3.1

torchvision >= 0.4.2

pandas >= 0.23.0

### Training

1. Download datatsets for FGVC (e.g. CUB-200-2011, NA-Birds, Standford Cars, FGVC-Aircraft, etc) and organize the structure as follows:
```
data
├── bird_train.txt
├── bird_text.txt
├── train
│   ├── Black_Footed_Albatross_0065_796068.jpg
|   ├── Black_Footed_Albatross_0042_796071.jpg
|   ├── Black_Footed_Albatross_0090_796077.jpg
│   └── ...
└── test
    ├── Black_Footed_Albatross_0046_18.jpg
    ├── Black_Footed_Albatross_0002_55.jpg
    ├── Black_Footed_Albatross_0085_92.jpg
    └── ...
```

2. Train from scratch with ``train.py``.


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- mazhanyu@bupt.edu.cn
- beiyoudry@bupt.edu.cn
