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
* Large batch methods can fail to generalize as well as small batch. 
    * Large batch methods tend to converge to sharp mimimizers of the training function.
    * Small batch methods tend to converge to flat minimizers.

### Binary Model Insights

* Output of many binary classification algorithms is a prediction score. The score indicates the certainty that an observation belongs to the positive class.
    * Output is a score that goes from 0 to 1.
    * Consumer of the score must decide the threshold that indicates the cut-off between 0 and 1. The default cut off is 0.5.
* Accuracy metric for binary classification: Area Under the (Receiver Operating Characteristic) Curve (AUC)

### Adjusting Class Probability to Increase Sensitivity

> The decision threshold adjustment was developed to estimate the optimal decision threshold for specified misclassification costs and/or prior probabilities of the prevalence [10, 11, 13]. When the class sizes are unequal, a shift in a decision threshold to favor the minority class can increase minority class prediction.
> ...
> In summary, the effect of class imbalance depends on the imbalance ratio, total sample size in the training phase, distributions of the data in each class and selection of the relevant variables as well as the classification algorithm and correction strategy.



### SMOTE

* Imbalanced datasets - a dataset is imbalanced if the classication categories are not approximately equally represented.
* Typically...
    * Real world data set includes only a small percentage of abnormal/interesting examples
    * Cost of misclassifying an abnormal/interesting example much higher than the reverse
* Combination of over-samplingthe minority (abnormal ) class and under-sampling the majority (normal ) class can achieve better classifier performance (in ROC space )than only under-sampling the majority class

Referenced on the sample quiz...

* Bootstrapping is any test or metric that uses random sampling with replacement, and falls under the broader class of resampling methods.

### Instance Segmentation

> Many of the companies rely on image segmentation techniques powered by Convolutional Neural Networks (CNNs), which form the basis of deep learning for computer vision. Image segmentation involves drawing the boundaries of the objects within an input image at the pixel level. This can help achieve object detection tasks in real-world scenarios and differentiate between multiple similar objects in the same image.

> Semantic segmentation can detect objects within the input image, isolate them from the background and group them based on their class. Instance segmentation takes this process a step further and can detect each individual object within a cluster of similar objects, drawing the boundaries for each of them.

### SageMaker API

### Kinesis Streaming Analytics ML Integration

### SageMaker Deployment and Lifecycle

* Inlcude canary weight ratios

### AWS Batch - Overview

### T-SNE

* t-Distributed Stochastic Neighbor Embedding
* Unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data
* PCA is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances
* t-SNE differs from PCA by preserving only small pairwise distances or local similarities whereas PCA is concerned with preserving large pairwise distances to maximize variance.

### SGD Optimization Techniques

Summary of excellent write up available [here](https://ruder.io/optimizing-gradient-descent/)

Gradient descent

* Technique to minimize an objective function by updating the parameters in the opposite direction of the gradient of the objective function.
* The learning rate determines the size of the steps we take to reach a (local) minimum

Variants (also see earlier in these notes)

* Batch, aka vanilla gradient descent
    * Computes the gradient of the cost function wrt the parameters of the entire training set

```
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

* Stochastic
    * Peforms an update for each training example

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

* Mini batch
    * Performs an update for each mini batch of n training examples

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

Challenges

* Choosing the learning rate. 
    * Too small - slow convergence
    * Too large - hinders convergence, may fluctuate around the minimum or event diverge
* Learning rate schedules try to adjust the learning rate during training by annealing, adjust the learning rate according to a predefined schedule or when the change in objective between epochs falls below a threshold
* Same learning rate for all parameter updates - might want larger updates for rarely occurring features
* Can get trapped in local minima/saddle points

Optimization Algorithms



### Factors the lead to the wide adoption of neural networks

* mag of data, algortirhms, checper gpus

Correlation strength

* positive is stronger, direction important not just magnitude

