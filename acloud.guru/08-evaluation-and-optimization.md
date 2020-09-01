# Evaluation and Optimization

## Intro

After training the model... is it any good?

## Evaluation and Optimization Concepts

Goal: we want generalization, not memorization.

Evaluation:

1. Define evaluation. Decide what metric or metrics we should use to decide if the algorithm is good enough.
2. Evaluate. Review the metrics during or after the training process. This might be manual or automatic, depending on the algorithm.
3. Tune. Adjust hyperparameters, data, the evaluation strategy or even the entire algoritm to bring us closer to the desired results.

Two Types:

* Offline Validation
    * Validation done using test sets of data
    * Example: validation sets and k-fold validation
* Online Validation
    * Validation under real-world conditions
    * Example: canary deployments

## Monitoring and Analyzing Training Jobs

Think feedback loops... training (test) metrics and validation metrics.

* SageMaker provides model training and validation metrics for each model, for example see [here](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html#linear-learner-metrics) for linear learner.

Algorithm metrics

* Algorithm runs in a container, logs are scraped to surface metrics in cloudwatch
* View in Sagemaker console or in the cloudwatch console
* Log entries available too
* Can do the same for custom algoritms via metric definitions using regex specs.

## Evaluating Model Accuracy

We crave generalization.

Underfitting

* Our model isn't reflective of the underlying data shape.
* Occurs when the model is too simple, i.e. informed by too few features or regularized too much.
* To prevent:
    * Add features
    * More training iterations

Overfitting

* Our model is too dependent on the data used to train the model. If it sees new data, accuracy will be poor unless it is identical to the training data.
* We have trained our algorithm to memorize rather than generalize.


Robust model

* Our model fits the training data but also does reasonably well on new data it has never seen.
* It can deal with noise in the data.
* It can generalize effectively for that new data.

| Training Error | Test Error | Comments |
| -- | -- | -- |
| Low | Low | You want this. |
| Low | High | Overfitting |
| High | High | Try another approach |
| High | Low | Run away! |

Preventing Overfitting

* More data. Sometimes more data will provide enough additional entropy to steer your algorithm away from overfitting.
* Early stopping. Terminate the training process before it has a chance to overtrain. Many algorithms include this option as a hyperparameter.
* Sprinkle in some noise. Your training data could be too clean and you might need to introduce some noise to generalize the model.
* Regularize. Regularization forces your model to be more general by creating constraints around weights or smoothing the input data.
* Ensembles. COmbine different models together to either amplify individual weaker models (boosting) or smoothing out stronger models (bagging)
* Ditch some features (i.e. feature selection). Too many irrelevant features can infuence the model in a negative way by drowning out the signal with noise.

### Regression Accuracy

* Residuals - a difference between the actual value and the predicted value. Negative residual meams prediction was higher than the actual.
* Histogram of residuals - want to center around 0.
* Can plot on a 2D coordinate - distance between the actual and the predicted is the residual.

RMSE - root mean square error (lower is better)

* Sum the square of the residuals
* Take the mean
* Take the square root of the mean of the redisuals

### Binary Classification Accuracy

| | Actual True | Actual False |
| -- | -- | -- |
| **predicted true** | correct prediction | false positive: TYPE I Error |
**predicted false**  | false negative: TYPE II error | correct prediction |



#### AUC - area under the curve

* For guaging binary classification accuracy
* Ranges from 0 to 1
* Want score as close to 1 as possible


#### F1 Score

* Recall = true positives / true positives + false negatives
    * Spam gets through
* Precision = true positives / true positives + false positives 
    * Legitimate email gets blocked
* Tradeoff - as one goes up the other goes down
* F1 score
    * 2 * (Precision * Recall) / (Precision + Recall)
    * The balance between Precision and Recall
    * A larger value indicates better predictive accuracy

### Multiclass Classification

Confusion matrix

## Improving Model Accuracy

* Collect datat. Increase the number of training example available to the model. More (good) data usually means a more accurate model.
* Feature processing. Provide additional quality variables or refine the existing variables so they are more representative.
* Model Parameter Tuning. Adjust the hyperparameters used by your training algorithm.
* Beware of bias and over-fitting.

