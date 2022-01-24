# data-analysis-python-ml
1.**Detecting Parkinson's Disease(fitting data into rule based algorithms)**

  *Data*  : Studying the correlation between speech metrics and the Parkinson's disease

  * * *
  *Steps followed for doing the analysis*:
  - Preprocessed the given data i.e. checked null values, datatypes & scaling of the sample data to the population by using Central Limit Theorem*
  - Fixed the label imbalance - `imblearn.RandomOverSampler.fit_resample(x,y)`
  - Scaled the data - `MinMaxScaler(-1,1).fit_transform(x i.e.  independent variables)`
  - Feature extraction and reduced the dimensions of data  (Principal component analysis**) - `PCA.fit_transform(x)`
  - Fit the data into test and train - `train_test_split(X_PCA, y, test_size=0.2, random_state=7)`
  - Applying algorithms  (*Logistic Regression, Decison Tree, Random Forest(information gain), Random Forest(Entropy), SVM, KNN, Gaussian naive bayes, Bernoulli naive bayes, voting classifier, XGBoost*) 
  `.fit().predict()`
  - Print the metrics and evaluation parameters of the models (classification reports, confusion matrix for false positive and false negative)
  - Plotting the **Receiver Operating Characteristic** curve. The further away the curve from the operating line(true positive vs false positive) the better the algorithm
  - Find out which model is best suited for the problem
  * * *
  *Conclusion*:

  - The best model to use in this case would be the Random forest algorithm as it has 0 false positives. Also it has the best auc score of 1.

  ***
  **The theorem states that as the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution.*

  ***PCA finds the eigenvectors of a covariance matrix with the highest eigenvalues and then uses those to project the data into a new subspace of equal or less dimensions. Practically, PCA converts a matrix of n features into a new dataset of (hopefully) less than n features*
