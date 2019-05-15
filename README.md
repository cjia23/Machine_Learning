# Machine_Learning

1. House_Price_prediction

Kaggle Project Links: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

Implemented 5 Neural Network Models, linear regression and SVM model with rbf kernels. Compared the different parameters for common neural network models, NO. of layers, NO. of neurons in hidden layer, activation function, optimizer(learning rate) and loss function.

2. Parkinson Diease Classification

Dataset: https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification

Compared the significance of 6 individule feature groups, also using Recursive Feature selection method to select the top 100 features. Implemented Neural Network, SVM linear, SVM rbf, Random Forest, Logistic Regression as well as two ensemble models. Achieved a highest accuracy rate of 72%, 0.71 F1 score and 0.70 recall score.


3. Anomaly detection - Steel surface tempreture

Implemented 6 gaussian distribution models in this anomaly detection assignment, concerning 
1. all features as independent 
2. most important 2 features. (I used chi2 test to find the top 2 features in the previous step.) 
3. PCA principle component analysis, projection of features into the top 2 features. 
Also, concerning using variance and co-variance. Thus resulting in 6 models. 

I also did supervised learning of SVB-rbf and clustering with K-means appraoches. 

Finally I tried to tune the epsilon values which can maximize the F1 score (a better indicator in this case than accuracy.) Further tuning will be required, basic thoughts are: get 1000 or 10,000.... epsilon values by slicing between mean to max of p values. Then store them into an python list. Find the epsilon with the biggest F1 score.

