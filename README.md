# Spring 2021, KAIST CS470 Course Project
A Replication of _Neural Predictor for Neural Architecture Search, ECCV'20_ with Tensorflow 2 [[Report](https://github.com/itouchz/Neural-Predictor-Tensorflow/blob/main/Team3_Final%20Report.pdf)] [[Presentation](https://github.com/itouchz/Neural-Predictor-Tensorflow/blob/main/Final%20Presentation.pdf)]

### File Descriptions
**Modules**

+ neural_predictor.py: implementaion of the neural predictor with its variants (MLP and CNN-based models).
+ search_spaces.py: functions for accessing NAS-Bench-101, ProxylessNAS, and NAS-Bench-NLP search spaces.
+ input_preprocessing.py: functions for preprocessing input with respect to each search space
+ random_search.py: random search methods for NAS-Bench-101 and ProxylessNAS

---
**Experiments**

+ Neural Predictor.ipynb: main experiments on neural predictor
+ Two-stage Predictor.ipynb: experiments of two-stage neural predictor (with classifer)
+ E1-NP-1.ipynb: neural predictor replication of Fig.4 in the original paper
+ E1-NP-2.ipynb: neural predictor replication of Fig.4 in the original paper
+ E1-NP-3.ipynb: neural predictor replication of Fig.4 in the original paper
+ E1-NP-4.ipynb: neural predictor replication of Fig.4 in the original paper
+ E1-Oracle.ipynb: Oracle replication of Fig.3 & 4 in the original paper
+ E1-Random-1.ipynb: Random search replication of Fig.4 in the original paper
+ E1-Random-2.ipynb: Random search replication of Fig.4 in the original paper
+ Ablation Study-1.ipynb: N vs K ablation study
+ Ablation Study-2.ipynb: different architecture ablation study
+ Extended Study.ipynb: extend experiments on NAS-Bench-NLP

---
**Directories**

+ figures: reproduced results from the original paper with a few additional figures
+ nasbench: original [NAS-Bench-101](https://github.com/google-research/nasbench) search space
+ nasbench_nlp: orignal [NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release) search space
+ proxylessnas: MobileNetv2-based ProxylessNAS search space
+ outputs: saved experimental results from the above experiments

