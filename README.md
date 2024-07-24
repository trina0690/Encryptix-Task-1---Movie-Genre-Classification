# 🎬 Movie Genre Classification

This project aims to build a machine learning model that predicts the genre of a movie based on its plot summary or other textual information. Techniques such as TF-IDF vectorization and classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) are used.

## 📁 Dataset

We use three datasets:
1. `train_data.txt`: Training data containing movie names, genres, and descriptions.
2. `test_data.txt`: Test data containing movie names and descriptions.
3. `test_data_solution.txt`: Solution for the test data containing movie names, genres, and descriptions.

   
Dataset link: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## 🛠️ Setup

### Prerequisites

- Python 3.6 or above
- Required libraries: pandas, numpy, sklearn, nltk, matplotlib, wordcloud

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/trina0690/Encryptix-Task-1---Movie-Genre-Classification.git

🚀 Steps
1. Data Loading and Preprocessing
Load training and testing datasets.
Preprocess the text data (convert to lowercase, remove punctuation, remove stopwords, apply stemming).
2. Exploratory Data Analysis (EDA)
Visualize the most common words in each genre using word clouds.
3. TF-IDF Vectorization
Convert text data to TF-IDF features to prepare for model training.
4. Model Training and Evaluation
Naive Bayes: Train a Multinomial Naive Bayes model and evaluate its performance.
Logistic Regression: Train a Logistic Regression model and evaluate its performance.
Support Vector Machine (SVM): Train an SVM model and evaluate its performance.
5. Model Comparison
Compare the accuracy and classification reports of Naive Bayes, Logistic Regression, and SVM models.
## 📊 Results:

- Multinomial Naive Bayes:
Accuracy: 0.52
Precision, recall, and f1-score metrics for each genre.


- Bernoulli Naive Bayes:
Accuracy: 0.52
Precision, recall, and f1-score metrics for each genre.


- Logistic Regression:
Accuracy: 0.59
Precision, recall, and f1-score metrics for each genre.


- Support Vector Machine (SVM):
Accuracy: 0.38
Precision, recall, and f1-score metrics for each genre.
The best model based on accuracy and classification report is Logistic Regression, with an accuracy of around 58.7%.

# 🙌 Acknowledgments

Kaggle for the dataset.

Encryptix for giving me this opportunity.
