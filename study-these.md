Some additional resources to study:

1. Use of metric to improve fraud detection.

> This is an example where the dataset is imbalanced with fewer instances of positive class because of a fewer number of actual fraud records in the dataset. In such scenarios where we care more about the positive class, then using PR AUC is a better choice, which is more sensitive to the improvements for the positive class.

> PR AUC is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot it. The higher on y-axis your curve is the better your model performance.

> Please review these excellent resources for a deep-dive into PR AUC.

> https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

> https://machinelearningmastery.com/imbalanced-classification-with-the-fraudulent-credit-card-transactions-dataset/

2. Viz for detecting outliers

> https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

> https://humansofdata.atlan.com/2017/10/how-to-find-outliers-data-set/

3. Specificity

https://www.statisticshowto.datasciencecentral.com/sensitivity-vs-specificity-statistics/

4. SageMaker high level python library

https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-batch-transform.html

5. SageMaker IAM

https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html

6. TF-IDF

https://en.wikipedia.org/wiki/Tf%E2%80%93idf

7. Inference pipelines

https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html

8. Personalize for Recommendations

music recommendations

https://aws.amazon.com/personalize/

9. Batch Processing Node Communications/Architecture

10. Cyclical Feature Engineering

http://blog.davidkaleko.com/feature-engineering-cyclical-features.html


11. IP Insights

12. Automatic Model Tuning

https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html

13. Early Stopping

> If the value of the objective metric for the current training job is worse (higher when minimizing or lower when maximizing the objective metric) than the median value of running averages of the objective metric for previous training jobs up to the same epoch, Amazon SageMaker stops the current training job.

14. Kinesis Video Streams

15. Network Isolation - Unsupported managed SageMaker containers

https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html

> Network isolation is not supported by the following managed Amazon SageMaker containers as they require access to Amazon S3

And these...

1. Linear learner hyper parameters, modes, inference outputs, use cases (iclude image classificatio)

2. General random forest algs

https://medium.com/datadriveninvestor/decision-tree-and-random-forest-e174686dd9eb and https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991

3. Data pipeline

https://docs.aws.amazon.com/datapipeline/latest/DeveloperGuide/what-is-datapipeline.html

4. Preprocessing in inference pipelines

https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

5. Hyperparameters

k-means

6. Sage Maker Validation

https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-model-validation.html

7. Ground Truth Labeling

https://aws.amazon.com/blogs/machine-learning/use-the-wisdom-of-crowds-with-amazon-sagemaker-ground-truth-to-annotate-data-more-accurately/

8. AWS IoT Analytics

https://aws.amazon.com/iot-analytics/

https://aws.amazon.com/blogs/big-data/build-a-visualization-and-monitoring-dashboard-for-iot-data-with-amazon-kinesis-analytics-and-amazon-quicksight/

9. Specifying Hyperparameter Tuning Job Settings

https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-tuning-job.html

10. Metrics Monitoring

https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/

https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-tuning.html

11. Using Tensorflow with SageMaker

https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html

12. XGBoost

https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html

13. Model tracking

https://aws.amazon.com/about-aws/whats-new/2019/08/new-model-tracking-capabilities-for-amazon-sagemaker-now-generally-available/

14. Lifecycle

https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_UpdateEndpoint.html

https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html

15. Batch Transform Pre and Post Processing

https://aws.amazon.com/about-aws/whats-new/2019/07/sagemaker-batch-transform-enable-associating-prediction-results-with-input-attributes/

16. ML transforms in AWS glue

https://docs.aws.amazon.com/glue/latest/dg/machine-learning.html

https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api-crawler-pyspark-extensions-dynamic-frame.html

17. Ref architecture - build and automate serverless data lake

https://aws.amazon.com/blogs/big-data/build-and-automate-a-serverless-data-lake-using-an-aws-glue-trigger-for-the-data-catalog-and-etl-jobs/

