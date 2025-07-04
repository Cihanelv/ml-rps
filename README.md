# Rock-Paper-Scissors Image Classification with CNN

This project is developed as part of the *Machine Learning Projects* course (A.Y. 2024/25) under the supervision of Prof. Nicolò Cesa-Bianchi at Università degli Studi di Milano.

---

##  Project Objective

To build and evaluate Convolutional Neural Network (CNN) models that classify images of hand gestures (rock, paper, scissors) using a sound machine learning pipeline:

* Exploratory data analysis and preprocessing
* Model architecture design and comparison
* Hyperparameter tuning via cross-validation
* Final evaluation and error analysis

---

## 📁 Directory Structure

```
ml-rps/
├── notebooks/        # Jupyter Notebooks
├── src/              # Python source modules (EDA, model, utils)
├── data_raw/         # Manual: Place archive/ folder here
├── data_split/       # Automatically created (train/val/test folders)
├── figures/          # Saved plots and images
├── requirements.txt  # Dependency list
└── README.md         # Project documentation
```

---

## Dataset Setup

This project uses the [Rock-Paper-Scissors dataset from Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).

###  Steps:

1. Download `archive.zip` from Kaggle.
2. Extract its contents into `ml-rps/data_raw/` → You should now have:

```
ml-rps/data_raw/archive/
├── rock/
├── paper/
└── scissors/
```

> ⚠️ The `notebooks/` code will fail if this structure is not respected.

---

## 🛠️ Installation

Make sure you have Python ≥ 3.8. Then run:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
numpy==1.26.0
pandas==2.3.0
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==1.7.0
scikeras==0.13.0
tensorflow==2.16.2
Pillow==11.2.1
imagehash==4.3.2
```

---

##  How to Run the Project

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the main notebook:

```
notebooks/rock-paper-scissors.ipynb
```

3. Run all cells (Shift + Enter or Run All)

> ⚠️ Make sure you have already downloaded and extracted the dataset as instructed.

---

## 📊 Pipeline Overview

### 1. **EDA & Data Splitting**

* Class distribution
* Sample images
* `train/val/test` stratified split

### 2. **Modeling**

* 3 CNN Architectures (Simple → Intermediate → Complex)
* Comparison via 3-fold cross-validation

### 3. **Hyperparameter Tuning**

* `RandomizedSearchCV` over key parameters
* Best configuration selected for final training

### 4. **Final Evaluation**

* Accuracy / Loss curves
* Classification report (Precision, Recall, F1)
* Confusion matrix
* Misclassified image visualization

---

## Reproducibility Notes

* Random seeds are fixed across NumPy, Python, and TensorFlow.
* Model evaluation uses **unseen test data**.
* All preprocessing and training is done without data leakage.

---

##  Author & Course Info

**Student:** Cihan Elveren
**Course:** Machine Learning Projects
**Instructor:** Nicolò Cesa-Bianchi
**Teaching Assistants:** Luigi Foscari, Emmanuel Esposito

---

##  License & Credits

* Dataset: [Kaggle RPS Dataset by DrGFreeman](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* Frameworks: TensorFlow, scikit-learn, scikeras
