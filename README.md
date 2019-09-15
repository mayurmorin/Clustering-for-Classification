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
   - Observations from describe function which explains count, mean, std, min and max values for each feature/variable
   - Shape is 1537 rows and 474 columns
   - Standardization makes each feature's mean to 0 and standard deviation to 1 which helps in calculation when applying different alogrigthms.
   - Train and Test split and standardized scaled Values stored in csv for further processing.
   - K-Means, Heirarchical Clustering and DBScan Cluster alogrithms evaluated for the given data.
   - K-Means and Heirarchical Clustering shows around 2-10 clusters whould be optimium to use for clustering algorithm.
   - DBSCAN algorithm does not show any clusters
   
3. In src Folder, there is a file called app.py.ipynb where first clustering and then classification algorithm have applied.
   - Libraries load
   - time function defined to calculate the time taken for execution
   - In EDA, it has been noticed DBSCAN was not performing was KMeans and HClust function have been made
   - New class clust have been created with KMeans and HClust Clustering algorithms and then classify function to classify the result
   - clust(load_data).Kmeans(output='add').classify() --> Uses Kmeans clustering with Logistic Regression 
     'Kmeans'  2453.68 ms
      Accuracy: 0.9025974025974026
      'classify'  9012.03 ms
   - clust(load_data).Kmeans(output='add').classify(SVC()) --> Uses Kmeans clustering with SVC Classification
      'Kmeans'  1975.90 ms
      Accuracy: 0.9329004329004329
      'classify'  2802.73 ms
   - clust(load_data).HClust(output='add').classify(SVC()) --> Uses Heirarchical Clustering with SVC Classification
      Accuracy: 0.9329004329004329
      'classify'  2567.15 ms
4. 
