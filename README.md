# Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN
## Overview
The code implements three popular machine learning classifiers—k-Nearest Neighbors (k-NN), Gaussian Naive Bayes, and Support Vector Machine (SVM)—to classify images based on extracted features. It preprocesses the data, normalizes it, and applies Principal Component Analysis (PCA) for dimensionality reduction. The code includes functions for calculating accuracy, precision, recall, F1 score, and confusion matrix. It evaluates each classifier's performance using both Euclidean distance and cosine similarity for k-NN. Additionally, the code visualizes results through line plots, 3D scatter plots, and confusion matrices, providing a comprehensive overview of each classifier’s effectiveness on the given dataset.

![Pastel Colorful Illustrative Digital Marketing Infographic (1)](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/0df34083-035f-48e9-8f82-ff0525f8b456)

# Pipline Explanation =
## 1) Data Loading:
- Import the dataset containing image features and labels.
 Loads the images and labels from the source, preparing for preprocessing.

## 2) Data Preprocessing:
- Clean and prepare the data for feature extraction.
- Converts images to a suitable format, handles missing values, and prepares data for feature extraction.
  
## 3) Feature Extraction:
- Extract numerical features from images.
-  Derives feature vectors from images for use in the classifiers.
-  
## 4) Normalization:
- Scale the feature vectors.
- Normalizes features to ensure they are on a similar scale for effective model performance.
  
## 5) Dimensionality Reduction (PCA):
- Reduce the number of features while retaining variance.
- Applies PCA to transform and reduce dimensionality, simplifying the feature space.
  
## 6) Splitting Dataset (Train/Test):
- Divide data into training and testing sets.
- Splits the dataset into training and testing subsets to evaluate model performance.
  
## 7) Model Selection:
- Choose the classifiers (k-NN, Gaussian Naive Bayes, SVM).
- Select k-NN, Gaussian Naive Bayes, and SVM for comparison.
  
## 8) Model Training:
- Train the models on the training dataset.
- Fits the classifiers on the training data, learning from features.

## 9) Model Evaluation:
- Evaluate the models using accuracy, precision, recall, F1 score, and confusion matrix.
- Assesses models’ performance on the test data with various metrics and similarity measures (Euclidean, cosine).

## 10) Results Visualization:
- Visualize the performance and metrics of each model.
- Generates line plots, 3D scatter plots, and confusion matrices to interpret and compare classifier effectiveness.
  
# Key Features
## k-Nearest Neighbors (k-NN):
- Supports both Euclidean and Cosine distance metrics.
- Evaluate different values of k for classification accuracy.
- Includes methods for accuracy, precision, recall, and F1 score evaluation.
  
  ### Euclidean Distance Metrics (for K=5) =
  
![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/d77f81d2-540f-48b6-a074-55124fcefa4a)

 ### Cosine Distance Metrics (for K=5) =
 
 ![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/a6d61598-459d-4a42-a35b-5c8ffebcc52e)

## Gaussian Naive Bayes (GNB):
- Assumes Gaussian distribution for features.
- Computes the likelihood of each class given the features.

## Support Vector Machine (SVM):
- Implements SVM with gradient descent optimization.
- Suitable for high-dimensional data classification.

## Data Preprocessing:
- Normalizes the dataset.
- Splits data into training and testing sets.
- Performs Principal Component Analysis (PCA) for dimensionality reduction.

## Visualization:
- Plots sample images with predicted and actual labels.
- Generates confusion matrices for performance evaluation.
  
  ![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/413c4211-b177-4b3a-b822-0c4ec3108f09)

- Visualizes the data in 3D space using PCA.
  
  ![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/80ad79e2-bf44-4206-8a88-ca257d792155)


 ![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/89ecac73-07c1-4139-a494-a621b793a2df)

 ![image](https://github.com/areeba0/Image-Classification-using-Gaussian-Naive-Bayes-SVM-and-KNN/assets/136759791/33169c70-e311-470d-b3ac-edce200708c9)




