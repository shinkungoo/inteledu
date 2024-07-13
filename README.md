<div align='center'>
<h1>InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis</h1>

<a href='https://aiedu.ecnu.edu.cn/'>Shanghai Institute of AI Education</a>, <a href='http://www.cs.ecnu.edu.cn/'>School of Computer Science and Technology</a>

East China Normal University

<img src='docs/inscd.svg' width=500 />
</div>

InsCD, namely Instant Cognitive Diagnosis (Chinese: 时诊), is a highly modularized python library for cognitive diagnosis in intelligent education systems. This library incorporates both traditional methods (e.g., solving IRT via statistics) and deep learning-based methods (e.g., modelling students and exercises via graph neural networks). 

<div align='center'>

<a href=''><img src='https://img.shields.io/badge/pypi-1.0.0-orange'></a> 
<a href=''><img src='https://img.shields.io/badge/Project-Page-brown'></a>
<a href=''><img src='https://img.shields.io/badge/Paper-PDF-yellow'></a>

</div>

## 📰 News 
- [x] [2024.7.14] InsCD toolkit v1.0 is released and available for downloading.

## 🚀 Getting Started
### Installation
Git and install with pip:
```
git clone https://github.com/ECNU-ILOG/inscd.git
cd [path of code]
pip install .
```
or install the library from pypi
```
pip install inscd
```
### Quick Example
The following code is a simple example of cognitive diagnosis implemented by inscd. We load build-in datasets, create cognitive diagnosis model, train model and show its performance:  
```python
from inscd.datahub import NeurIPS20
from inscd.models.neural import NCDM

listener.activate()
datahub = NeurIPS20()
datahub.random_split()
datahub.random_split(source="valid", to=["valid", "test"])

ncdm = NCDM()
ncdm.build(datahub)
ncdm.train(datahub, "train", "valid")

test_results = ncdm.score(datahub, "test", metrics=["acc", "doa"])
```

For more details, please refer to **[InsCD Documentation](https://sites.google.com/view/inscd-doc/home)**.


## 🤗 Contributor
Contributors are arranged in alphabetical order by first name. We welcome more people to participate in maintenance and improve the community of intelligent education.

Junhao Shen, Mingjia Li, Shuo Liu, Xin An, Yuanhao Liu

## Citation
If this repository is helpful and can inspire you in your reseach or applications, please kindly cite as follows.

### BibTex
```

```

### ACM Format

