# 📱 SMS Spam Classification using Natural Language Processing (NLP)

This repository contains the code and resources for the **SMS Spam Classification** project, which uses **Natural Language Processing (NLP)** techniques to classify SMS messages as spam or non-spam. The project employs several machine learning algorithms, including **Linear Support Vector Machine (SVM)**, **Multinomial Naive Bayes**, and **Binary Logistic Regression**, to build and evaluate classification models for effective spam detection.

## 📂 Repository Contents

- 📘 **LogisticRegression.ipynb**: Jupyter Notebook containing the implementation of the Logistic Regression model for spam classification.
- 📗 **NaiveBayes.ipynb**: Jupyter Notebook with the code for the Multinomial Naive Bayes classifier, a probabilistic approach to identify spam messages.
- 📕 **SupportVectorMachine.ipynb**: Jupyter Notebook implementing the Linear Support Vector Machine (SVM) model, which finds the optimal hyperplane for categorizing messages.
- 📊 **spam.xlsx**: Excel file containing the dataset used for training and testing the models. This dataset includes SMS messages labeled as spam or non-spam.

## 📝 Project Overview

The primary objective of this project is to develop a robust SMS spam filtering system using various NLP techniques and machine learning algorithms. The system preprocesses the text data, extracts features, and applies multiple classification models to accurately detect spam messages.

### ✨ Key Steps in the Project

1. **📥 Import Essential Libraries**: Import necessary libraries like NumPy, Pandas, Matplotlib, and others for data manipulation, visualization, and model development.
2. **📄 Import Dataset**: Load the dataset containing SMS messages labeled as spam or ham (non-spam).
3. **🧹 Data Cleaning**: Clean the data by removing noise, non-English characters, null values, and duplicate sentences to standardize the text.
4. **🔍 Data Exploration and Analysis**: Analyze the dataset by calculating SMS text length, understanding class distribution, and visualizing data through bar and pie charts.
5. **✂️ Tokenization**: Break down text into individual tokens (words) and visualize the result using bar charts and WordCloud.
6. **🌿 Lemmatization**: Apply lemmatization to reduce words to their base form, ensuring uniformity (e.g., "running" becomes "run").
7. **🔢 Vectorization**: Convert text data into numerical format using Count Vectorization, representing the frequency of words.
8. **🏷️ Label Encoding**: Encode the labels "spam" and "ham" into numerical format (1 and 0, respectively).
9. **📊 Data Preparation**: Split the dataset into training (80%) and test (20%) subsets to prepare for model training.
10. **🤖 Model Training**: Train machine learning models (Linear SVM, Multinomial Naive Bayes, and Binary Logistic Regression) on the training data.
11. **🧪 Model Testing**: Test the model on both training and test datasets to evaluate its ability to generalize to unseen data.
12. **📉 Model Evaluation**: Evaluate model performance using metrics such as accuracy, precision, recall, confusion matrix, and learning curve analysis.
13. **🚀 Model Implementation**: Deploy the model for practical use, ensuring it performs effectively in real-world scenarios.

## 🚀 Getting Started

To use this repository, clone it to your local machine and open the Jupyter Notebooks to explore the code. Ensure you have the necessary Python libraries installed, such as `scikit-learn`, `pandas`, `numpy`, and `nltk`.

## 📊 Evaluation Metrics

- **Learning Curves**: To observe the model's learning performance over time.
- **Accuracy, Precision, Recall**: To measure the model's effectiveness in classifying messages.
- **Confusion Matrix**: To provide a visual representation of the classification results.

Feel free to explore the notebooks and experiment with the models to enhance their performance!
