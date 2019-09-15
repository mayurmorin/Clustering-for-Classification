# Clustering-for-Classification

#Explanation on Folder Structure:
1. Data : the raw data for which classification is required to be done.
2. notebooks : Feature Engineering and Exploratory Data Analysis
3. models : Trying different Models
4. src : Model Selection

#Feature Engineering and Exploratory Data Analysis:
1. In notebooks folder, there is a file called feature engineering which does following activities:
   - Read data from Training Dataset.
   - Shows datatypes of each feature/variable
   - Count Number of records/row in the dataset
   - differentiate and give count of categorical and numerical features
   - drop categorical feature because of it's uniqueness like an id
   - remove features which are uniqune or have only 1 value like all the values are 0,1, or 100.
   - dealing with missing values using Imputer
   - reduced fatures from 590 to 474 using this process
   - Saved cleaned data into data cleaning.csv for further processing.
   - Check Feature Enginerring.ipynb and Images (Feature Engineering_1.png,Feature Engineering_2.png,Feature Engineering_3.png) for   reference.
   
2. In notebooks folder, there is a file called exploratory data analysis which does following activities:
