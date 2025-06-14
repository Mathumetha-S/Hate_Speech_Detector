# Hate_Speech_Detector
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-ML/NLP-purple)]()
[![Framework](https://img.shields.io/badge/Built%20with-Scikit--learn-orange)](https://scikit-learn.org/)
[![Language](https://img.shields.io/badge/Languages-Hindi%2C%20Tamil-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](./LICENSE)
[![Requirements](https://img.shields.io/badge/Requirements-pandas%2C%20sklearn%2C%20transformers-blue)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

## Tamil Hate Speech Detector🗣️
**Tamil Hate Speech Detector** is a specialized machine learning project built to automatically detect hate or offensive speech in Tamil text. Using Natural Language Processing (NLP) techniques combined with multilingual transformer models (bert-base-multilingual-cased) this project aims to classify Tamil user-generated content into normal, offensive, or hateful categories.

It is designed to help platforms and communities moderate Tamil regional language content and maintain a safe digital environment.

This project is intended for researchers, developers, and students who want to build ethical AI systems that address the increasing problem of cyberbullying and online hate speech in Tamil.

## 🚀 Features

- 🧠 **Tamil Language Hate Speech Detection**
  - Focused exclusively on Tamil regional language content.
- ⚡ **State-of-the-art Transformer Model**
  - Utilizes **bert-base-multilingual-cased**, a multilingual transformer model fine-tuned for Tamil hate speech classification.
- 📚 **Dataset**
  - Uses real-world **Tamil datasets** from **Kaggle** and other research sources.
- 📊 **Model Evaluation**
  - Provides detailed evaluation using:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **Confusion Matrix**
- 🧪 **Research-ready**
  - Suitable for:
    - **Academic research**
    - **AI/ML projects**
    - **Thesis/dissertation work**
- ♿ **Impact**
  - Supports **Tamil content moderation**.
  - Helps promote **safe, respectful, and inclusive online spaces** for Tamil-speaking users.


## 🛠 Tech Stack

| 💡 Area                     | 🧰 Tools & Libraries                                           |
| --------------------------- | -------------------------------------------------------------- |
| 🐍 **Programming Language** | Python 3.10+                                                   |
| 📚 **Data Handling**        | `pandas`, `numpy`                                              |
| 🤖 **Machine Learning**     | `scikit-learn`, `transformers`                                 |
| 🧠 **NLP Model**            | `bert-base-multilingual-cased`  (Hugging Face Transformers)                    |
| 📊 **Visualization**        | `matplotlib`, `seaborn`                                        |
| 📝 **Development Tools**    | Jupyter Notebook, VS Code, Google Colab                        |
| 🗃️ **Dataset**             | **Tamil Hate Speech Dataset** (from Kaggle & Research Sources) |

## 📁 Folder Structure


tamil-hate-speech-detector/
├── datasets/
│   └── tamil_offensive_speech_train.csv
│   └── tamil_offensive_speech_val.csv
├── tamil_hate_speech_detector.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── troubleshooting.txt

## 📈 Model Overview


| **Component** | **Description**                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Tokenizer** | **bert-base-multilingual-cased Tokenizer** (Supports Tamil text tokenization)                                                       |
| **Model**     | **bert-base-multilingual-casedForSequenceClassification** fine-tuned **only on Tamil hate speech dataset**                           |
| **Training**  | Trained **exclusively on Tamil hate speech datasets**                                                              |
| **Metrics**   | **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Confusion Matrix** for Tamil text classification accuracy |


## 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

