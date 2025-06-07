# Hate_Speech_Detector
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-ML/NLP-purple)]()
[![Framework](https://img.shields.io/badge/Built%20with-Scikit--learn-orange)](https://scikit-learn.org/)
[![Language](https://img.shields.io/badge/Languages-Hindi%2C%20Tamil-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](./LICENSE)
[![Requirements](https://img.shields.io/badge/Requirements-pandas%2C%20sklearn%2C%20transformers-blue)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

A multilingual machine learning project to detect hate speech in Hindi and Tamil using NLP and XLM-RoBERTa.
**Hate Speech Detector** is a multilingual machine learning project designed to automatically detect hate or offensive speech in text written in **Hindi** and **Tamil**. This project uses NLP techniques and multilingual transformer models (XLM-RoBERTa) to classify user-generated content, helping online platforms and communities moderate regional language content.

This solution is intended for researchers, developers, and students building ethical AI systems to combat cyberbullying and hate speech in diverse languages.

## 🚀 Features

- 🧠 Multilingual classification for Hindi and Tamil
- 📊 Model training using XLM-RoBERTa transformer
- 📚 Real-world datasets from Kaggle and research sources
- 🧪 Evaluation metrics: Accuracy, F1-Score, Confusion Matrix
- ♿ Useful for content moderation & digital safety


| 💡 Area                 | 🧰 Tools & Libraries                             |
| ----------------------- | ------------------------------------------------ |
| 🐍 Programming Language | Python 3.10+                                     |
| 📚 Data Handling        | `pandas`, `numpy`                                |
| 🤖 Machine Learning     | `scikit-learn`, `transformers`                   |
| 🧠 NLP Model            | `XLM-RoBERTa` (HuggingFace Transformers)         |
| 📊 Visualization        | `matplotlib`, `seaborn`                          |
| 📝 Development Tools    | Jupyter Notebook, VS Code, Google Colab          |
| 🗃️ Dataset             | Hindi & Tamil hate speech datasets (from Kaggle) |

## 📁 Folder Structure
hate-speech-detector/
├── data/
│ ├── Hatespeech-Hindi_Train.csv
│ ├── Hatespeech-Hindi_Valid.csv
│ └── tamil_offensive_speech_train.csv
├── models/
│ └── xlm-roberta/
├── hate_speech_classifier.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── troubleshooting.txt

##📈 Model Overview

Tokenizer: XLM-RoBERTa Tokenizer
Model: XLMRobertaForSequenceClassification
Training: Fine-tuned with multilingual hate speech data
Metrics: Accuracy, Precision, Recall, F1

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

