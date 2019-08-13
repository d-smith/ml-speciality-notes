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


### Types of Machine Learning

Supervised Learning

* Learning with feedback provided in the learning data
* Each training example is provided with the correct label
* Regression - target type is a continuous value
* Classification - target type is catagorical

Unsupervised Learning

* No target column/outcode
* Have a collection of features/attributes from each observation
* Grouping/clustering for downstream analysis

Reinforcement Learning

* Algorithm is not told what action to take, but is given a reward or penalty for after each action in a sequence
* Example - teach a machine how to play video games

### Key Issues in Machine Learning

Data Quality

* High quality data is the secret sauce
* Quality
    * Consistency of the data - consider the business problem; is the data we're using consistent with the problem we are trying to solve
    * Accuracy of the data - labels, features (numerical and categorical)
    * Noisy data - many input and output fluctuations
    * Missing data - algoritms can't deal with missing data
    * Outliers - errors, typos, correct but not relevant or out of scope
    * Bias
    * Variance


Model Quality

* Underfitting vs overfitting
* Overfitting
    * Failure to generalize - performs well on training data but poorly on test
    * Can indicate model is too flexible
    * Flexible - allows it to memorize the data including the noise
    * Corresponds to high variance - small changes in the training data leads to big changes in the results
* Underfitting
    * failure to capture important patterns in the training data set
    * Typically indicates model is too simple or there are too few explanitory variables
    * Not flexible enough to model real patterns
    * Corresponds to high bias - the results show systematic lack of fit in certain regions

Computation Speed and Scalability

* Use distributed computing systems like Sage Maker of EC2 instances for training in order to:
    * Increase speed
    * Solve prediction time complexity
    * Solve space complexity
* May need to address latency and scalability instances

### Supervised Methods: Linear Regression

#### Linear regression

Linear methods

* Parametric methods where function learned has form f(x) = &Phi;(w<sup>T</sup>x) where &Phi;() is some activation function.
* Generally optimized by learning weights by applying (stochastic) gradient descent to minimize loss function e.g. &Sigma; |y<sup>&Hat;</sup><sub>i</sub> - y<sub>i</sub>|<sup>2</sup>
* Simple; a good place to start for a new problem, at least as a baseline
* Methods
    * linear regression for numeric target outcome
    * logistic regression for categorical target outcome

Univariate linear regression

* Model relation between a single feature (explanatory variable x) and a real-valued response (target variable y), for example area as price predictor for real estate price
* Error is |y<sup>&Hat;</sup><sub>i</sub> - y<sub>i</sub>|, e.g. predicted price minus real price.
* Given data (x,y) and a line defined by w<sub>0</sub> (intercept) and slope w<sub>1</sub> (slope), the vertical offset for each data point from the line is the error between the true label y and the prediction based on x
* The best line minimizes the sum of squared errors (SSE)
* We usually assume the error is Guassian distributed with mean zero and fixed variance
