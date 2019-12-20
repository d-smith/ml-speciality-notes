# K-Fold Cross Validation

Relatively small data set - want to fully use the data set

Issue: Small sets

* Smaller training set - not enough data for good training
* Unrepresentative test set - invalid metrics

K-Fold Cross-Validation

* Randomly partition data into K folds
* For each fold, train model on other k-1 folds and evaluate on that
* Train on all data
* Average metric across k-folds estimate test metric for trainined model

K-Fold CV: Choosing K

* Large - more time, more variance
* Small - more bias
* Typical value of k is 5 - 10

Leave-one-out cross-validation

* K = number of data points
* Used for very small sets

Stratified K-fold cross validation

* Preserve class proportions in the folds
* Use for imbalanced data
* There are seasonality or subgroups