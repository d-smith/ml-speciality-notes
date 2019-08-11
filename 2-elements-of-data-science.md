# The Elements of Data Science

This course covers:

* Intro to data science and machine learning
* Problem formultation, data collection, exploratory data analysis
* Data preprocessing and feature engineering
* Model training, tuning, and debugging
* Model evaluation, productionizing a ML model


Ref: symbols via html entities see [here](https://html.spec.whatwg.org/entities.json). Markdown reference [here](https://github.github.com/gfm/#inlines)

## What is Data Science?

### Introduction to Data Science

What is data science?

* General definition: processes and systems to extract knowledge or insights from data, either structured or unstructured.
* For the purposes of this course: managing, analyzing, and visualizing data in support of machine learning workflows.

What is machine learning?

* Artificial intelligence machines that improve their predictions by learning from large amounts of input data.

Machine Learning

* Main idea: learning equals estimating underlying function f by mapping data attributes to some target value
* Training set: a set of labeled examples (x, f(x)) where x is the input variable and the label f(x) is the observed target truth
* Goal: given a training set, find approximation f<sup>&Hat;</sup> of f that best generalizes, or predicts, labels for new examples.
    * Best is measured by some quality measure
    * Example: error rate, sum squared error

Why Use Machine Learning

* Some problems are too difficult to solve using conventional programming techniques
    * Too complex (facial recognition)
    * Too much data (stock market predictions)
    * Information only available dynamically
* Use of data for improvement
    * Humans are used to improving based on experience
* A lot of data is available
    * Product recommendations
    * Fraud detection
    * Facial recognition
    * Language understanding

Types of machine learning algoritms

* Supervised learning - target variable used to determine the truth. Human intelligence baked in with the labeling of the training data.
* Unsupervised learning - only a collection of features with no specific outcome, no human in the loop
* Semi-supervised learning - combination of supervised and unsuperised learning, used in cases where you have a mixture of labeled and unlabeled data.
* Reinforcement learning - you use rewards and penalties to let the algoritm learn, reward outcomes and behaviors you want to encourage.

Data Matters

* The more data, the better the accuracy
* High quality training data set is the essential component for success

Data Science Workflow

* Phase 1 - Problem Formation
* Phase 2
    * Data Collectopm
    * Exploratory Analysis
    * Data Preprocessing
    * Feature Engineering
* Phase 3
    * Model training
    * Model evaluation
    * Model tuning
    * Model debugging
* Phase 4
    * Productionisation

Important Concepts

* The data set - partition, don't use the same data in training and model validation
* The validation set should be targeted towards business requirements, but include enough outliers to give a realistic view of the model's performance.
* Feature = attribute = independent variable = predictor
* Label = target = outcome = class = dependent variable = response
* Dimensionality = number of features
* Model selection


