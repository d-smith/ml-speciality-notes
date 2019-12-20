# Cross Validation

Issue: metrics on training data can't measure generalization

* Model could cheat by memorizing the data and getting a perfect score
* Overfitting

Solution:

* Cross validation: train and evaluate on distinct data sets

Holdout Method

* Split data set into separate training and test sets
    * Training setL used to train, tune, and select model
    * Test set: used to evaluate final model

`sklean.model_selection - train_test_split`

