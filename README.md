<div align='center'>
<h1>InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis</h1>

<a href='https://aiedu.ecnu.edu.cn/'>Shanghai Institute of AI Education</a>, <a href='http://www.cs.ecnu.edu.cn/'>School of Computer Science and Technology</a>

East China Normal University

<img src='docs/inscd.svg' width=700 />
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
cd <path of code>
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

## 🛠 Implementation
We incoporate classical, famous and state-of-the-art methods accepted by leading journals and conferences in the field of psychology, machine learning and data mining. The reason why we call this toolkit "modulaized" is that we not only provide the "model", but also divide the model into two parts (i.e., extractor and interaction function), which enables us to design new models (e.g., extractor of Hypergraph with interaction function of KaNCD). To evaluate the model, we also provide vairous open-source datasets in online or offline scenarios.

### List of Models
|Model|Release Time|Paper|
|-----|------------|-----|
|Item Response Theory (IRT)|||
|Multidimentional Item Response Theory (MIRT)|||
|Deterministic Input, Noisy "And" Gate (DINA)|||
|Neural Cognitive Diagnosis Model (NCDM)|||
|Item Response Ranking (IRR)|||
|Knowledge-association Neural Cognitive Diagnosis (KaNCD)|||
|Knowledge-sensed Cognitive Diagnosis Model (KSCD)|||
|Q-augmented Causal Cognitive Diagnosis Model (QCCDM)|||
|Relation Map-driven Cognitive Diagnosis Model (RCD)|||
|Hypergraph Cognitive Diganosis Model (HyperCDM)|||

### List of Build-in Datasets
|Dataset|Release Time|Paper|
|-------|------------|-----|
||||

## 🤔 Frequent Asked Questions
> Why I cannot download the dataset when using build-in datasets class (e.g., `NeurIPS20` in `inscd.datahub`)?

Since these datasets are saved in the  Google Driver, they may be not available in some countries and regions. You can use proxy and add the following code before using build-in datasets.
```
os.environ['http_proxy'] = 'http://<IP address of proxy>:<Port of proxy>'
os.environ['https_proxy'] = 'http://<IP address of proxy>:<Port of proxy>'
os.environ['all_proxy'] = 'socks5://<IP address of proxy>:<Port of proxy>'
```

## 🤗 Contributor
Contributors are arranged in alphabetical order by first name. We welcome more people to participate in maintenance and improve the community of intelligent education.

Junhao Shen, Mingjia Li, Shuo Liu, Xin An, Yuanhao Liu

## 🗞 Citation
If this toolkit is helpful and can inspire you in your reseach or applications, please kindly cite as follows.

### BibTex
```

```

### ACM Format

