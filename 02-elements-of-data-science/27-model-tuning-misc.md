# More on Bias, Feature Extraction vs Feature Selection

## Bias

Bias is the estimation of the difference between the
fitted model and the actual relationship of response
and features.

A high bias model usually indicates lack of fit, i.e.
the model is not flexible enough to capture the
relatonship between the response variable and the
explanatory variables.

Solutions : more complicated models, adding features
(new features, transformed features,interactions between features)

## Feature Extraction and Feature Selection

Both feature extraction and selection are used to reduce
the diminsionality of the feature space.

Feature selection uses algorithms to send to some of
the features from the model such that the selected
features enable the model to perform better and there is
no change to the features.

Feature extraction uses algorithms to combine features
( using linear or non-linear transformations of the original
features for example) to generate a set of new
features, and the number of new features to be used
in the model is generally less than the orignal
number of features.