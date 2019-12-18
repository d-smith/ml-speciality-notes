# Model Tuning : Bagging /Boosting

Bagging and boosting are automated or semi- automated
approaches to determining which features to include

Bagging ( Bootstrap Aggregation )

Motivation : generate a group of weak learners that
when combined together generate
higher accuracy

* Create x datasets of size m by randonly sampling original dataset with splacement (duplicates allowed)
* Can apply to tree sclated models as wellas
* Any of the data points not selected for the
dataset can be used for the validation set
* Train weak learners (decision stumps,
logistic regression) on the new datasets
to generate predictions
* High Variance but low bias? => Use bagging 

Training many models on random subsetsof the data
and coverage / vote on the output

* Reduces variance
* Keeps bias the same
* sklearn sklearn.ensemble.BaggingClassifer, BaggingRegressor

Boosting - another ensemble method

* Assign strengths to each weak learner
* Iteratively train learners using misclassifed examples
by the previous weak learners
* Model has a high bias and accepts weights an individual samples? =? Use boosting

Training a sequence of samples to get a strong model
often wins on datasets like most Kaggle competitions

sklearn.ensemble.AdaBoostClassifire,Regressor, Gradient Boosting Classifier

XGBoost library
