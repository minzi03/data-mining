# data-mining
# Heart Disease Prediction

This project involves predicting the likelihood of heart disease based on a variety of medical and lifestyle features using machine learning algorithms. The dataset used for this project contains demographic, medical, and lifestyle information, with the target variable being the presence or absence of heart disease.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering and Selection](#feature-engineering-and-selection)
6. [Modeling](#modeling)
7. [Model Evaluation](#model-evaluation)
8. [Deployment](#deployment)
9. [Requirements](#requirements)
10. [License](#license)

### Project Overview

The main goal of this project is to build a machine learning model that can predict the presence of heart disease using a variety of features, including lifestyle habits, physical health metrics, and demographic data. This project performs data analysis, preprocessing, feature selection, model building, and evaluation to provide the best predictive model.

### Data Overview

The dataset used in this project, Heart Disease.xlsx, contains the following features:

- **Smoking**: Whether the person is a smoker.
- **AlcoholDrinking**: Whether the person drinks alcohol.
- **Stroke**: Whether the person has experienced a stroke.
- **DiffWalking**: Difficulty in walking.
- **Sex**: Gender of the person.
- **Race**: The race of the person.
- **PhysicalActivity**: Whether the person engages in physical activity.
- **SkinCancer**: Whether the person has skin cancer.
- **BMI**: Body Mass Index.
- **PhysicalHealth**: Number of days the person’s physical health was not good in the past month.
- **MentalHealth**: Number of days the person’s mental health was not good in the past month.
- **SleepTime**: Average sleep time per day.
- **HeartDisease**: Target variable indicating whether the person has heart disease or not.

### Exploratory Data Analysis (EDA)

In the EDA phase, the dataset is explored visually and numerically to understand the distributions of features and identify potential correlations. Here are the steps performed:

- **Class Distribution**: Visualized the distribution of HeartDisease (target) and other features, such as Smoking, AlcoholDrinking, Stroke, and DiffWalking.
- **Correlation Heatmap**: Visualized the correlation matrix to understand the relationships between features and the target variable.
- **Pairwise Feature Distributions**: Used Kernel Density Estimation (KDE) plots to visualize the distribution of features with respect to heart disease.
- **Unbalanced Data**: Observed that the dataset is imbalanced, with more negative cases than positive heart disease cases.

### Data Preprocessing

To prepare the data for machine learning, the following steps were performed:

- **Label Encoding**: Categorical variables such as HeartDisease, Smoking, Sex, Race, etc., were label-encoded to convert them into numerical values.
- **Handling Imbalanced Data**: To address the class imbalance in the target variable, undersampling was performed using the NearMiss algorithm.

### Feature Engineering and Selection

Feature selection techniques were applied to select the most important features for model training:

- **Feature Importance with ExtraTreesClassifier**: We used the ExtraTreesClassifier to assess the importance of each feature based on the model's decision trees.
- **SelectKBest**: We used the SelectKBest algorithm with the f_classif method to score features and select the best ones.
- **PCA (Principal Component Analysis)**: PCA was applied to reduce the dimensionality of the dataset and retain the most significant components.

### Modeling

Several machine learning models were trained and evaluated for heart disease prediction:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- AdaBoost Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Each model was trained and evaluated using accuracy, classification report, confusion matrix, and ROC-AUC score.

### Model Evaluation

After training the models, their performance was evaluated on the test set. The evaluation metrics used are:

- **Accuracy**: Measures the percentage of correctly predicted instances.
- **Confusion Matrix**: Visualized the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
- **ROC Curve and AUC**: The Receiver Operating Characteristic (ROC) curve was plotted to visualize the performance of the classifiers. The Area Under the Curve (AUC) was calculated for each model.

### Deployment

After determining the best-performing model, the model was deployed using the Gradio library to create a web interface where users can input their medical and lifestyle features and receive a prediction of whether they are at risk for heart disease.

The function `heart` takes multiple input features and outputs a prediction for heart disease. The web interface allows users to input these features interactively.
