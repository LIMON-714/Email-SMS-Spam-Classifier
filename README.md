# Email-SMS-Spam-Classifier
A Machine Learning-based Email/SMS Spam Classifier using TF-IDF Vectorization and Voting Classifier (SVC, Multinomial Naive Bayes, and Extra Trees). Built with Python and Streamlit to detect spam and legitimate messages with an interactive web interface.

# 📩 Email/SMS Spam Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-purple?style=for-the-badge" alt="NLP">
</p>

<h3 align="center">
  An Intelligent Machine Learning Application for Email and SMS Spam Detection.
</h3>

<p align="center">
  Detect Spam and Not Spam messages using Natural Language Processing and Ensemble Machine Learning.
</p>

<p align="center">
  <img width="1271" height="618" alt="image" src="https://github.com/user-attachments/assets/cf59e7da-a249-4e49-81e9-33067b1431f9" />
</p>>

---

## 📌 About The Project

The **Email/SMS Spam Classifier** is a Machine Learning-based web application designed to classify text messages into two categories:

- 🚨 **Spam:** Unwanted, promotional, or potentially fraudulent messages.
- ✅ **Not Spam:** Legitimate and normal messages.

This project combines Natural Language Processing (NLP), TF-IDF Vectorization, and a Soft Voting Classifier to analyze and classify messages.

The application is developed using Python and Streamlit, providing an interactive interface where users can enter a message and instantly receive a classification result.

## ✨ Key Features

- 📩 Email and SMS Spam Detection
- 🤖 Machine Learning-based Classification
- 🧠 Natural Language Processing (NLP)
- 🔤 Text Cleaning and Tokenization
- 🚫 Stopword Removal
- 🌱 Porter Stemming
- 📊 TF-IDF Feature Extraction
- 🎯 Soft Voting Classifier
- 📈 Spam Probability Estimation
- 🌐 Interactive Streamlit Interface
- 🧹 Clear Message Function
- 🔍 Processed Message Visualization

## 🖥️ Application Screenshots

### Code Implementation

<p align="center">
  <img width="1365" height="696" alt="image" src="https://github.com/user-attachments/assets/1efd40a6-e0eb-46ec-a13d-112cf8879b51" />
</p>

### 🚨 Spam Detection Result

<p align="center">
  <img width="1309" height="600" alt="image" src="https://github.com/user-attachments/assets/ab7d6a3c-9df3-4b08-b29d-dea14dc798a5" />
</p>

### ✅ Not Spam Detection Result

<p align="center">
  <img width="1301" height="620" alt="image" src="https://github.com/user-attachments/assets/d849313d-4e30-4502-bc57-64a24af96a3a" />
</p>


## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Main Programming Language |
| Pandas | Dataset Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| NLTK | Natural Language Processing |
| TF-IDF | Text Feature Extraction |
| Streamlit | Web Application |
| Pickle | Model Serialization |
| Jupyter Notebook | Model Training and Experimentation |

## 🤖 Machine Learning Algorithms

This project uses a **Soft Voting Classifier** combining three different machine learning algorithms.

### 1. Support Vector Classifier (SVC)

SVC is used to classify messages by finding an optimal decision boundary between spam and legitimate messages.

### 2. Multinomial Naive Bayes

A probabilistic algorithm widely used for text classification based on word frequency and probability.

### 3. Extra Trees Classifier

An ensemble learning algorithm that combines multiple randomized decision trees to improve classification performance.

### 4. Soft Voting Classifier

The Soft Voting Classifier combines the predicted probabilities from the individual models and produces the final classification.

## 🔬 Natural Language Processing Workflow

The application processes the input message using the following steps:

1. Convert text to lowercase.
2. Tokenize the message.
3. Remove stopwords.
4. Remove unnecessary punctuation and symbols.
5. Apply Porter Stemming.
6. Transform processed text using TF-IDF.
7. Pass the features to the trained Voting Classifier.
8. Generate the final prediction.

## 🔄 Project Workflow

```text
              Input Message
                    |
                    v
            Text Preprocessing
                    |
                    v
               Tokenization
                    |
                    v
             Stopword Removal
                    |
                    v
              Porter Stemming
                    |
                    v
            TF-IDF Vectorization
                    |
                    v
             Voting Classifier
                    |
          +---------+---------+
          |                   |
          v                   v
        SPAM               NOT SPAM
          |                   |
          v                   v
     Spam Result         Safe Result
```

## 📂 Project Structure

```text
Email-SMS-Spam-Classifier/
│
├── app.py
│
├── model.pkl
├── vectorizer.pkl
│
├── data/
│   └── spam.csv
│
├── screenshots/
│   ├── home.png
│   ├── spam-result.png
│   └── not-spam-result.png
│
├── Spam_Classifier.ipynb
│
├── README.md
│
└── requirements.txt
```

### 📁 Folder Description

| File / Folder | Description |
|---|---|
| `app.py` | Streamlit web application |
| `model.pkl` | Trained Voting Classifier model |
| `vectorizer.pkl` | Saved TF-IDF Vectorizer |
| `data/spam.csv` | SMS Spam dataset |
| `screenshots/` | Application screenshots |
| `Spam_Classifier.ipynb` | Model training and experimentation |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |

## 📊 Dataset Information

This project uses the **SMS Spam Collection Dataset**, available on Kaggle.

| Dataset Details | Information |
|---|---|
| Dataset Name | SMS Spam Collection Dataset |
| Source | Kaggle |
| Original Repository | UCI Machine Learning Repository |
| File Name | `spam.csv` |
| File Location | `data/spam.csv` |
| Classification Type | Binary Classification |
| Categories | Ham and Spam |

### 📥 Download Dataset

The dataset can be downloaded from Kaggle:

<p align="center">
  <a href="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset">
    <img src="https://img.shields.io/badge/Download%20Dataset-Kaggle-blue?style=for-the-badge&logo=kaggle" alt="Download Dataset">
  </a>
</p>

**Dataset Link:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

After downloading, place the CSV file inside the `data/` directory.

```text
Email-SMS-Spam-Classifier/
└── data/
    └── spam.csv
```

The dataset contains SMS messages classified into two categories:

- `ham` – Legitimate messages.
- `spam` – Unwanted or unsolicited messages.

## 🚀 Installation & Setup

```

###  Install Required Libraries

```bash

```

### Run the Streamlit Application

```bash
streamlit run app.py
```

### Open the Application

After running the command, open the following address in your browser:

```text
http://localhost:8501
```

## 📦 Requirements

The project dependencies are listed in `requirements.txt`.

```text
streamlit
pandas
numpy
scikit-learn
nltk
```

## 🎯 Project Objectives

The main objectives of this project are:

- To understand and implement Natural Language Processing techniques.
- To perform text preprocessing and feature extraction.
- To build an ensemble-based Machine Learning classification model.
- To detect spam and legitimate messages.
- To deploy a Machine Learning model using Streamlit.
- To develop an interactive and user-friendly web application.

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Add model performance visualization.
- Include confusion matrix and classification reports.
- Support email file uploads.
- Improve the user interface and user experience.
- Deploy the application online.

## 👨‍💻 Author

### MD LIMON RAHMAN

**Machine Learning | Python | Data Science | NLP**

---

<p align="center">
  ⭐ If you find this project useful, please consider giving it a star!
</p>

<p align="center">
  Made with ❤️ using Python, Machine Learning and Streamlit.
</p>
