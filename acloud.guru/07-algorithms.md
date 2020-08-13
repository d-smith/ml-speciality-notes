# Algorithms

Still in the train the model part of the machine learning cycle.

## Algorithm Concepts

Definition: unambiguous specification of how to solve a class of problems.

* An algorithm is a set of steps to follow to solve a specific problem intended to be repeatable with the same outcome.
* Contrast with heuristic, which is a mental short-cut or rule of thumb that provides guidance on doing a task but does not guarantee a consistent outcome.

Algorithms in Machine Learning

* Drive out bias by avoiding hueristics
* Still need to look out for bias - data we select for training or testing, or exclude an important chunk of sample data
* Feedback loop can introduce bias - might assume we'll see a certain set of results

SageMaker

* Use built in algorithms
* Purchase from AWS marketplace
* Build your own via docker image

## Regression

Linear Learner Algorithm

* Linear models are supervised learning algorithms for regression, binary classification or multiclass classification problems. You give the model labels (x,y) with x being a high dimensional vector and y as a numeric label. The algoritm learns a linear function, or, for classification problems, a linear threshold function, and maps a cector x to an approximation of  label y.
* To use this algorithm you need a number or list of numbers which yields some other number - the answer you are after. You can use it to predict a specific value or a threshold for grouping purposes.

Adjust to Minimize Error

* Algorithm wants the equation to be as good of a fit as possible, meaning the sum of all the distances from the training data point to the fitted line is a small as possible.
* Stochastic Gradient Descent is used to minimize error.
    * Local and global minimums

Linear Learner for Classification

* Map text values a number representation
* Convert data to vectors

Linear Learner Characteristics

* Very flexible. Linear learner can be used to explore differnt training objectives and chhose the best one. Well suited for discrete or continuous inferences.
* Built-in Tuning. Linear learner algorithm has an internal mechanism for tuning hyperparameters separate from the automatic model tuning feature.
* Good first choice. If your data and objective meets the requirement, linear learner is a good first choice to try for your model.

Usage

* Predict quantitative value based on given numeric input
    * Example: based on the last 5 year ROI from marketing spend, proedice this years ROI
* Discrete binary classification problem
    * Example: based on past customer response, should I email this particular customer? Yes or no..
* Discrete multiclass classification problems
    * Example: based on past customer response, how shouldI reac hthis customer? Email, direct mail, phone call?

Sparse Data

* Linear learner works well with a large amount of contiguous data
* Dealing with sparse data: factorization machines

Factorization Machines Algorithm

* General purpose supervised learning algorithm for both binary classification and regression. Captures interaction between features with high dimensional sparse datasets.
* To use this algorithm you need a number or list of numbers which yields some other number - the number you are after. You can use it to predict a specific value or a threshold for placing into one of two groups. It is a good choice when you have holes in your data.
* Use:
    * When you have high dimensional spare data sets
        * Example: click stream data on which ads on a webpage tend to be clicked given known information about the person viewing the page.
    * Recommendations
        * Example: what sort of movies should we recommend to a person who has watched and rated some other movies

Things to know about factorization machines

* Considers only pairwise features - SageMaker's implementation will only analyze relationships of two pairs of features at a time.
* CSV is not supported. File and pipe mode training is supported using recordio-protobuf format with Float32 tensors.
* Doesn't work for multi-class problems. Binary classification and linear regression modes only.
* Needs LOTS of data. Recommended dimension of the input feature space is between 10,000 and 10,000,000.
* AWS recommends CPUs with factorization machines for the most efficient experience.
* Don't perform well on dense data.

Example: Movie recommendations

* Think of a 1/0 per movie title - lots of movie titles means sparse populaton of review per user
