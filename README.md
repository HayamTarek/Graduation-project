# Alzheimer Disease Prediction 
  Graduation-project
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
  * scikit-learn
  * matplotlib
  * seaborn
## Machine Learning Proposed Model
![image](https://github.com/HayamTarek/Graduation-project/assets/125991048/0bc9dff6-f852-4610-b059-d6f96efe5ca3)
## Model Comparison
![image](https://github.com/HayamTarek/Graduation-project/assets/125991048/5250bd29-66eb-41ed-84e9-d5f2539f99a1)
