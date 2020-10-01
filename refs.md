# Some Additional Refs

## References

* Pipe Mode - CSV Data Set - see [here](https://aws.amazon.com/blogs/machine-learning/now-use-pipe-mode-with-csv-datasets-for-faster-training-on-amazon-sagemaker-built-in-algorithms/)
* TF-IDF - see [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [Article on batches and training](https://arxiv.org/pdf/1609.04836.pdf)
* Binary Model Insights - see [here](https://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html)
* Adjusting class probability to increase sensitivity at the trade off of precision - see the Discussion section of [this](https://academic.oup.com/bib/article/14/1/13/304457) paper.
* SMOTE - [this](https://www.jair.org/index.php/jair/article/view/10302)
* Imputing missing data - [this](https://www.hilarispublisher.com/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.pdf)
* Handling missing values for classification - page 1627 [here](https://jmlr.csail.mit.edu/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf), and see [this](https://www.annualreviews.org/doi/10.1146/annurev.publhealth.25.102802.124410) and [this](https://docs.aws.amazon.com/machine-learning/latest/dg/feature-processing.html).
* SageMaker [object2vec](https://aws.amazon.com/blogs/machine-learning/introduction-to-amazon-sagemaker-object2vec/)

## Reference Notes

### TF-IDF

See the tf-idf.ipynb notebook in this directory for details on how to count unigrams, bigrams, etc.

### Batches and Training

Gradient Descent

* Gradient - slope or slant of a surface
* Gradient descent - descending a slope to find its lowest point
    * GD is an iterative algorithm that starts on a random point on a function and travels down its slope in steps until it finds the lowest point of that function.
* In GD, learning rate is the step size for adjusting parameters values as we move down the gradient.
    * Learning rate heavily influences the convergence of the algorithm. Too big can jump across a minimum.
* Can be computationally expensive 
    * Sum of squared residuals consists of as many terms as there are data points
    * Need to compute the derivative for each of the features which the number of data points times the number of features

Stochastic Gradient Descent

* Stochastic - means random. 
* Reduces computation by selecting one point at random in each step to use to compute the derivatices.

Mini-Batch

* Seeks to balance precision of GD vs speed of SGD by sampling a small number of data points at each step.


### SMOTE

* Imbalanced datasets - a dataset is imbalanced if the classication categories are not approximately equally represented.
* Typically...
    * Real world data set includes only a small percentage of abnormal/interesting examples
    * Cost of misclassifying an abnormal/interesting example much higher than the reverse
* Combination of over-samplingthe minority (abnormal ) class and under-sampling the majority (normal ) class can achieve better classifier performance (in ROC space )than only under-sampling the majority class

Referenced on the sample quiz...

* Bootstrapping is any test or metric that uses random sampling with replacement, and falls under the broader class of resampling methods.

