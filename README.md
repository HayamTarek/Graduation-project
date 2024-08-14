# Alzheimer Disease Prediction 
Alzheimer’s disease (AD) is Serious mental illness affecting the brain. the reason for dementia in the elderly people. Around 45 million people are suffering from this disease. Alzheimer’s disease spreads rapidly over the years. This disease has a significant impact on the social, financial, and economic aspects. Early treatment of Alzheimer's disease is more effective and causes less brain damage. Since Alzheimer's is a chronic disease, we should have taken care of early detection using machine learning. This proposed model represents the analysis and the result regarding detecting Dementia from various machine learning models. longitudinal Magnetic Resonance Imaging (MRI) data from OASIS has been used for the development of the system. Several techniques such as Decision Tree, Random Forest, Support Vector Machine, Logistic Regression and keras neural network have been employed to identify the best parameters for Alzheimer’s disease prediction. we obtained 0.97, 1.00, 0.944 and 0.971 for the Accuracy, precision, Recall, F1_score respectively.
## Project Overview
This project aims to develop a machine learning model capable of accurately predicting Alzheimer's disease based on longitudinal Magnetic Resonance Imaging (MRI) data. The model leverages various classification algorithms and preprocessing techniques to achieve optimal performance.
## Team Members
- Hayam Tarek 
- Asmaa Abdelfattah 
- Nada Shaban
- Shrouk Saeed
## Data and Preprocessing
The project utilizes MRI data from the OASIS dataset. The data undergoes the following preprocessing steps:
  * Categorical Data Conversion: Categorical data is transformed into numerical format for model compatibility.
  * Handling Missing Values: Any null or missing values within the dataset are addressed appropriately.
  * Feature Selection: Unnecessary columns are removed to enhance model efficiency.
  * Correlation Analysis: Correlation matrix is applied to identify potential relationships between features.
  * Data Splitting: Stratified sampling is employed to divide the data into training (80%) and testing (20%) sets.
## Methodology
The project explores the following classification algorithms:
  * Support Vector Machine (SVM)
  * Logistic Regression
  * Decision Tree
  * Random Forest
Each algorithm is trained and evaluated using the preprocessed dataset. Cross-validation with k-fold technique is applied to assess model robustness.
## Results and Evaluation
The Random Forest model demonstrates the highest accuracy of 97.33% among the tested algorithms. The other models achieved accuracies of:
  * SVM: 96%
  * Logistic Regression: 94.67%
  * Decision Tree: 96%
## Project Structure
The project repository is organized as follows:
  * data: Contains the preprocessed MRI data.
  * models: Stores trained machine learning models.
  * notebooks: Jupyter notebooks for data exploration and model development.
  * src: Python source code for the project.
## Dependencies
The project relies on the following Python libraries:
  * pandas
  * numpy
  * matplotlib
  * sklearn
## Machine Learning Proposed Model
![image](https://github.com/HayamTarek/Graduation-project/assets/125991048/0bc9dff6-f852-4610-b059-d6f96efe5ca3)
## Model Comparison
![image](https://github.com/HayamTarek/Graduation-project/assets/125991048/5250bd29-66eb-41ed-84e9-d5f2539f99a1)
