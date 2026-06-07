# 机器学习大作业 Gesticulate 实验报告
2023012134 王振宇
本项目已开源![Github仓库](https://github.com/Andromeda1101/Gesticulate)
---

## 研究背景

---

## 实验环境
1. 模型训练机
 - WSL Ubuntu 24.04
 - Python3.12 & Miniconda

2. 实时系统机
 - Windows 11
 - Python3.12 & Miniconda
 - 内置摄像机

---

## 实验设计

---

## 数据准备与预处理
### 数据集概况

#### 数据集同步相关说明
由于两个数据集标签有不同，在此处我们将数据集 `LeapGestRecog` 的标签向 `HaDRID` 标签对齐。
具体地：
| LeapGestRecog | HaDRID | 备注 |
| --- | --- | --- |
| 01_palm 08_palm_moved | stop | `LeapGestRecog` 中的 palm 为并指，形态特征更类似于 `HaDRID` 中的 stop |
| 02_l | thumb_index | 大拇指+食指 |
| 03_fist 04_fist_moved | fist | - |
| 05_thumb | like | 实际上`LeapGestRecog` 中的 thumb 为向右，`HaDRID` 中不存在类似动作 |
| 06_index | one | - |
| 07_ok | ok | - |
| 09_c | grip | - |
| 10_down | palm | - |

### 实验步骤

---

## 特征设计
### Keypoint

### HOG

### Keypoint + HOG

---

## 算法实现
### KNN

### Decision Tree

### Random Forest

### Naive Bayes

### Logistic Regression

### SVM

### MLP

### CNN

### LSTM

---

## 指标设计
### EXP-01 & EXP-02

### EXP-03

---

<!-- ## 实验 EXP-01
### 实验步骤

### 实验结果

模型对比实验（EXP-01）：在 **hybrid** 特征与统一划分协议下，比较各分类器在验证集上的性能（共 9 组 completed run，主指标 accuracy）。

| 算法 | accuracy | f1_macro | recall_macro | precision_macro | fit_seconds | inference_seconds | per_sample_inference_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | 0.8786 | 0.8801 | 0.8793 | 0.8844 | 8.0049 | 0.0349 | 0.0159 |
| lstm | 0.8772 | 0.8774 | 0.8770 | 0.8811 | 8.4692 | 0.0460 | 0.0209 |
| mlp | 0.8568 | 0.8614 | 0.8580 | 0.8720 | 6.7858 | 0.0693 | 0.0315 |
| random_forest | 0.8518 | 0.8621 | 0.8470 | 0.8863 | 313.7350 | 0.3636 | 0.1654 |
| logistic_regression | 0.8199 | 0.8236 | 0.8198 | 0.8401 | 14.9142 | 0.0030 | 0.0013 |
| knn | 0.8186 | 0.8187 | 0.8169 | 0.8328 | 0.1225 | 0.7715 | 0.3509 |
| naive_bayes | 0.8117 | 0.8207 | 0.8292 | 0.8325 | 0.0480 | 0.6431 | 0.2924 |
| decision_tree | 0.8008 | 0.8127 | 0.7952 | 0.8426 | 236.3971 | 0.0647 | 0.0294 |
| svm | 0.7299 | 0.7475 | 0.7240 | 0.8856 | 559.0951 | 1.7226 | 0.7834 |

## 实验 EXP-02
### 实验步骤

### 实验结果

特征消融实验（EXP-02）：固定算法、比较 **keypoints_only / hog_only / hybrid** 三种特征族在验证集上的 accuracy

| 算法 | keypoints_only | hog_only | hybrid |
| --- | --- | --- | --- |
| cnn | 0.7758 | 0.5184 | 0.8786 |
| lstm | 0.8654 | 0.4537 | 0.8772 |
| mlp | 0.8772 | 0.6227 | 0.8568 |
| random_forest | 0.8663 | 0.2545 | 0.8518 |
| knn | 0.8622 | 0.6071 | 0.8186 |
| decision_tree | 0.8131 | 0.2224 | 0.8008 |
| naive_bayes | 0.7576 | 0.5337 | 0.8117 |
| logistic_regression | 0.7326 | 0.5306 | 0.8199 |
| svm | 0.5484 | 0.6125 | 0.7299 |

各特征族下 accuracy 最高的算法：

| 特征族 | 最优算法 | accuracy |
| --- | --- | --- |
| keypoints_only | mlp | 0.8772 |
| hog_only | mlp | 0.6227 |
| hybrid | cnn | 0.8786 | -->

## 实验 EXP-01 $\times$ EXP-02
由于实验一与实验二本质相同，因此在此报告中我们将两个实验共同汇报
### 实验步骤

### 实验结果
综合实验 1 与实验 2 ，得到完整 算法-特征 表格如下：
| algorithm | feature_family | accuracy | f1_macro | recall_macro | precision_macro | f1_micro | recall_micro | precision_micro | fit_seconds | inference_seconds | per_sample_inference_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | hybrid | 0.8729 | 0.8761 | 0.8740 | 0.8839 | 0.8729 | 0.8729 | 0.8729 | 8.3713 | 0.0415 | 0.0188 |
| mlp | keypoints_only | 0.8697 | 0.8743 | 0.8706 | 0.8856 | 0.8697 | 0.8697 | 0.8697 | 8.1144 | 0.0239 | 0.0108 |
| lstm | hybrid | 0.8683 | 0.8703 | 0.8697 | 0.8759 | 0.8683 | 0.8683 | 0.8683 | 9.3860 | 0.0532 | 0.0241 |
| mlp | hybrid | 0.8525 | 0.8528 | 0.8539 | 0.8557 | 0.8525 | 0.8525 | 0.8525 | 8.1166 | 0.0367 | 0.0166 |
| lstm | keypoints_only | 0.8520 | 0.8580 | 0.8533 | 0.8710 | 0.8520 | 0.8520 | 0.8520 | 7.2829 | 0.0342 | 0.0155 |
| knn | keypoints_only | 0.8520 | 0.8532 | 0.8530 | 0.8598 | 0.8520 | 0.8520 | 0.8520 | 0.0059 | 1.0953 | 0.4956 |
| random_forest | keypoints_only | 0.8516 | 0.8609 | 0.8524 | 0.8842 | 0.8516 | 0.8516 | 0.8516 | 98.4930 | 0.3243 | 0.1468 |
| random_forest | hybrid | 0.8425 | 0.8519 | 0.8430 | 0.8749 | 0.8425 | 0.8425 | 0.8425 | 349.1048 | 0.3982 | 0.1802 |
| logistic_regression | hybrid | 0.8249 | 0.8257 | 0.8253 | 0.8395 | 0.8249 | 0.8249 | 0.8249 | 41.2438 | 0.0118 | 0.0054 |
| naive_bayes | hybrid | 0.8222 | 0.8279 | 0.8236 | 0.8502 | 0.8222 | 0.8222 | 0.8222 | 0.0487 | 0.9088 | 0.4112 |
| naive_bayes | keypoints_only | 0.8167 | 0.8222 | 0.8191 | 0.8481 | 0.8167 | 0.8167 | 0.8167 | 0.0106 | 0.0483 | 0.0218 |
| knn | hybrid | 0.8136 | 0.8138 | 0.8149 | 0.8257 | 0.8136 | 0.8136 | 0.8136 | 0.0271 | 2.2378 | 1.0126 |
| logistic_regression | keypoints_only | 0.7905 | 0.7889 | 0.7912 | 0.8170 | 0.7905 | 0.7905 | 0.7905 | 6.3392 | 0.0021 | 0.0009 |
| cnn | keypoints_only | 0.7855 | 0.7899 | 0.7874 | 0.8023 | 0.7855 | 0.7855 | 0.7855 | 8.6127 | 0.0290 | 0.0131 |
| decision_tree | keypoints_only | 0.7783 | 0.7996 | 0.7782 | 0.8501 | 0.7783 | 0.7783 | 0.7783 | 25.0579 | 0.0041 | 0.0019 |
| decision_tree | hybrid | 0.7724 | 0.7938 | 0.7727 | 0.8451 | 0.7724 | 0.7724 | 0.7724 | 199.5153 | 0.0047 | 0.0021 |
| svm | hybrid | 0.7724 | 0.7969 | 0.7730 | 0.8958 | 0.7724 | 0.7724 | 0.7724 | 688.5034 | 2.5304 | 1.1450 |
| svm | keypoints_only | 0.6543 | 0.6830 | 0.6531 | 0.8733 | 0.6543 | 0.6543 | 0.6543 | 638.2898 | 0.6929 | 0.3135 |
| mlp | hog_only | 0.6282 | 0.6290 | 0.6282 | 0.6413 | 0.6282 | 0.6282 | 0.6282 | 6.0272 | 0.0352 | 0.0138 |
| knn | hog_only | 0.6141 | 0.6154 | 0.6141 | 0.6512 | 0.6141 | 0.6141 | 0.6141 | 0.0422 | 3.5444 | 1.3900 |
| logistic_regression | hog_only | 0.5663 | 0.5586 | 0.5663 | 0.5915 | 0.5663 | 0.5663 | 0.5663 | 35.4605 | 0.0093 | 0.0036 |
| naive_bayes | hog_only | 0.5431 | 0.5498 | 0.5431 | 0.5795 | 0.5431 | 0.5431 | 0.5431 | 0.0457 | 0.9438 | 0.3701 |
| cnn | hog_only | 0.5161 | 0.5175 | 0.5161 | 0.5402 | 0.5161 | 0.5161 | 0.5161 | 9.4443 | 0.0426 | 0.0167 |
| lstm | hog_only | 0.4420 | 0.4406 | 0.4420 | 0.4513 | 0.4420 | 0.4420 | 0.4420 | 10.9547 | 0.0436 | 0.0171 |
| random_forest | hog_only | 0.4243 | 0.4156 | 0.4243 | 0.4559 | 0.4243 | 0.4243 | 0.4243 | 338.8106 | 0.4171 | 0.1636 |
| svm | hog_only | 0.3773 | 0.4778 | 0.3773 | 0.8734 | 0.3773 | 0.3773 | 0.3773 | 1102.6826 | 2.4216 | 0.9497 |
| decision_tree | hog_only | 0.1882 | 0.2104 | 0.1882 | 0.2999 | 0.1882 | 0.1882 | 0.1882 | 260.9365 | 0.0130 | 0.0051 |

最好的三个结果的混淆矩阵如下：
![CNN+hybrid](./asserts/experiments/EXP-02_cnn_hybrid_confusion.png)
![MLP+keypoints_only](./asserts/experiments/EXP-02_mlp_keypoints_only_confusion.png)
![LSTM+hybrid](./asserts/experiments/EXP-02_lstm_hybrid_confusion.png)

### 实验分析

## 实验 EXP-03
### 实验步骤

### 实验结果

跨数据集鲁棒性实验（EXP-03）：在 HaGRID 子集上训练的 EXP-02 模型，于域内 test 与 LeapGestRecog（OOD）上评估；下表为全部特征下各模型的 OOD 对比。

| algorithm | feature_family | in_domain_accuracy | ood_accuracy | absolute_accuracy_drop | relative_performance_retention | ood_shared_subset_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| knn | hybrid | 0.8124 | 0.3955 | 0.4169 | 0.4868 | 0.3955 |
| logistic_regression | hybrid | 0.8097 | 0.3124 | 0.4972 | 0.3859 | 0.3124 |
| knn | hog_only | 0.6217 | 0.2210 | 0.4007 | 0.3555 | 0.2210 |
| mlp | hybrid | 0.8444 | 0.2207 | 0.6237 | 0.2613 | 0.2207 |
| logistic_regression | hog_only | 0.5598 | 0.1796 | 0.3802 | 0.3208 | 0.1796 |
| naive_bayes | hog_only | 0.5512 | 0.1727 | 0.3785 | 0.3133 | 0.1727 |
| mlp | hog_only | 0.6096 | 0.1698 | 0.4398 | 0.2785 | 0.1698 |
| lstm | hybrid | 0.8561 | 0.1490 | 0.7072 | 0.1740 | 0.1490 |
| random_forest | hog_only | 0.4253 | 0.1296 | 0.2957 | 0.3047 | 0.1296 |
| logistic_regression | keypoints_only | 0.7970 | 0.1293 | 0.6678 | 0.1622 | 0.1293 |
| knn | keypoints_only | 0.8561 | 0.1236 | 0.7325 | 0.1443 | 0.1236 |
| mlp | keypoints_only | 0.8620 | 0.1224 | 0.7396 | 0.1420 | 0.1224 |
| naive_bayes | hybrid | 0.8209 | 0.1218 | 0.6991 | 0.1484 | 0.1218 |
| lstm | hog_only | 0.4379 | 0.1199 | 0.3180 | 0.2738 | 0.1199 |
| cnn | keypoints_only | 0.7808 | 0.1051 | 0.6757 | 0.1346 | 0.1051 |
| cnn | hog_only | 0.5006 | 0.1035 | 0.3971 | 0.2068 | 0.1035 |
| lstm | keypoints_only | 0.8539 | 0.0968 | 0.7570 | 0.1134 | 0.0968 |
| cnn | hybrid | 0.8611 | 0.0850 | 0.7761 | 0.0987 | 0.0850 |
| random_forest | hybrid | 0.8412 | 0.0777 | 0.7635 | 0.0924 | 0.0777 |
| random_forest | keypoints_only | 0.8543 | 0.0765 | 0.7778 | 0.0895 | 0.0765 |
| decision_tree | hog_only | 0.1862 | 0.0702 | 0.1161 | 0.3767 | 0.0702 |
| decision_tree | hybrid | 0.7898 | 0.0547 | 0.7351 | 0.0693 | 0.0547 |
| naive_bayes | keypoints_only | 0.8205 | 0.0534 | 0.7671 | 0.0651 | 0.0534 |
| decision_tree | keypoints_only | 0.7866 | 0.0502 | 0.7364 | 0.0639 | 0.0502 |
| svm | hybrid | 0.7682 | 0.0460 | 0.7222 | 0.0599 | 0.0460 |
| svm | hog_only | 0.3889 | 0.0120 | 0.3769 | 0.0307 | 0.0120 |
| svm | keypoints_only | 0.6495 | 0.0029 | 0.6466 | 0.0044 | 0.0029 |

OOD accuracy 为全类评估；共享类 OOD accuracy 仅统计 HaGRID 与 LeapGestRecog 共有的 7 类手势。

### 实验分析

## 实验 EXP-04
### 实验步骤

### 实验结果

---

## 实验总结

