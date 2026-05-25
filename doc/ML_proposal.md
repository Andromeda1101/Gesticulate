# Research Proposal: Design and Implementation of a Machine Learning-Based Visual Gesture Recognition System for Keyboard Control

## 1. Introduction and Project Overview

Traditional Human-Computer Interaction (HCI) devices, such as physical keyboards and mice, pose operational constraints in touchless environments, smart offices, and assistive technologies. This project outlines the end-to-end development of a low-latency, computer-vision-powered gesture recognition system. By leveraging ordinary webcams, the system extracts hand geometry features, classifies intent using optimized machine learning pipelines, and maps inputs to keyboard events via native OS level APIs.

---

## 2. Feature Engineering & Algorithmic Framework

### 2.1 Feature Extraction Pipeline

The system utilizes two distinct types of representations to balance geometric precision with global spatial context:

1. 
**Geometric/Kinematic Features (Local):** * Mediapipe Hands is utilized to extract $N=21$ global 3D hand joint coordinates: $\mathbf{P}_i = (x_i, y_i, z_i)$.


* **Relative Distance Matrix:** Euclidean distances between all pairs of keypoints to ensure translation invariance:

$$D_{ij} = \|\mathbf{P}_i - \mathbf{P}_j\|_2$$


* 
**Joint Angles:** Cosine angles calculated across major bone linkages to capture finger flexion:



$$\theta = \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2}\right)$$


* 
**Min-Max Feature Scaling (Normalization):** Coordinates are normalized relative to the wrist $\mathbf{P}_0$ and scaled by the hand bounding box diagonal $L$ to enforce scale invariance:



$$\mathbf{P}'_i = \frac{\mathbf{P}_i - \mathbf{P}_0}{L}$$




2. **Hand Texture/Appearance Features (Global):**
* 
**Histogram of Oriented Gradients (HOG):** Extracted over the cropped hand bounding box to provide edge-gradient representation robust against slight illumination variations.





### 2.2 Classification Algorithms

To systematic explore the performance trade-offs, the coding agent will implement and optimize the following algorithms:

```
                       [cite_start]┌── K-Nearest Neighbors (KNN) [cite: 14]
                       [cite_start]├── Support Vector Machine (SVM) [cite: 14]
                       [cite_start]├── Decision Tree & Random Forest [cite: 14]
[cite_start]机器学习算法分类 ───────┼── Naive Bayes & Logistic Regression [cite: 14]
                       [cite_start]├── Multi-Layer Perceptron (MLP) [cite: 14]
                       [cite_start]└── Deep Sequences (CNN / LSTM) [cite: 14]

```

---

## 3. Dataset & Experimental Setup

### 3.1 Data Corpora

* 
**Primary Dataset (LeapGestRecog):** 10 categories of clean, static gestures (e.g., *Palm, Fist, Thumb Up, L-Shape, OK, Peace*). Used for core model training and baseline evaluation.


* 
**Generalization Dataset (HaGRID Subset):** Introduced as an out-of-distribution (OOD) target. Features complex backgrounds, varying lighting conditions, and diverse user groups to stress-test model robustness.



### 3.2 Data Split & Validation

The baseline dataset is partitioned into **70% Training, 15% Validation, and 15% Testing** subsets. The agent will execute a **Stratified 5-Fold Cross-Validation** during training to alleviate overfitting and stabilize hyperparameter tuning.

### 3.3 Coding Agent Action Items & Experimental Matrix

The code must be structured modularly to complete four core automated experiments:

| Exp ID | Experiment Type | Implementation Target | Focus Metric |
| --- | --- | --- | --- |
| **EXP-01** | <br>**Model Comparison** 

 | Benchmark all 9 algorithms under identical data splits.

 | Accuracy, Inference Speed, Training Wall-time 

 |
| **EXP-02** | <br>**Feature Ablation** 

 | Compare model behavior given (a) Keypoints-only, (b) HOG-only, (c) Hybrid Concatenated vectors.

 | Dimension reduction efficiency vs Accuracy 

 |
| **EXP-03** | <br>**Robustness Test** 

 | Train on LeapGestRecog $\rightarrow$ Zero-shot Test directly on HaGRID.

 | OOD Generalization Drop ($\Delta$ Accuracy) 

 |
| **EXP-04** | <br>**Real-Time Deployment** 

 | Deploy the champion model using OpenCV camera loops and event mapping.

 | End-to-end Latency, FPS stability 

 |

---

## 4. System Architecture & Deployment Realization

### 4.1 Real-Time System Pipeline

The runtime inference system operates as a continuous multi-threaded processing loop:

```
[Camera Video Stream] ──(OpenCV)──> [Frame Extraction] ──> [MediaPipe Tracker]
                                                                  │
[OS Keyboard Trigger] <──(PyAutoGUI)── [Champion Model] <── [Feature Vector]

```

### 4.2 Control Mapping Specification

The operational target maps physical gestures to the virtual keystrokes below:

* 
**Palm Gesture** $\longrightarrow$ `Key.space` (e.g., Presentation Pause / Video Play) 


* 
**Fist Gesture** $\longrightarrow$ `Key.enter` (e.g., Confirmation action) 


* 
**Thumb Up** $\longrightarrow$ `Key.up` (e.g., Volume Up / Page Up) 


* 
**Peace Sign** $\longrightarrow$ `Key.down` (e.g., Volume Down / Page Down) 



---

## 5. Evaluation Metrics & Expected Benchmarks

### 5.1 Quantitative Metrics

The agent must generate evaluation scripts compiling the following statistical attributes:

* 
**Classification Quality:** Macro/Micro Precision, Recall, Macro-F1 Score, and Multi-class Confusion Matrices.


* 
**Operational Performance:** Model Inference Delay (milliseconds) and overall Pipeline Execution Speed (Frames Per Second - FPS).



### 5.2 Target Benchmarks

* Discriminative classifiers (SVM, Random Forest) and standard deep structures (MLP) should cross the **$\ge$ 90% Accuracy threshold** on the static validation partition.


* The pipeline must operate sustainably at **$\ge$ 30 FPS** under standard consumer hardware webcams to ensure unnoticeable control latency.