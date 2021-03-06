# Notes From Practive Exams

AWS Sample Questions

https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf

ACG Main Exam

From here - https://learn.acloud.guru/course/aws-certified-machine-learning-specialty/dashboard


> You are a machine learning specialist finding ways to detect anomalous data points within a given labeled data set. You've been tasked with creating a model to achieve this and also determine how accurate the model is along with other metrics like precision, recall, and F1-score metrics on the labeled data. How can this easily be achieved?

Create a model using the Random Cut Forest (RCF) algorithm with both a train and the optional test data channels. Use text/csv for training and validation data. Train the model on an ml.m4 or ml.c4 instance type.

Amazon SageMaker Random Cut Forest (RCF) is an unsupervised algorithm for detecting anomalous data points within a data set. When using RCF the optional test channel is used to compute accuracy, precision, recall, and F1-score metrics on labeled data. Train and test data content types can be either application/x-recordio-protobuf or text/csv and AWS recommends using ml.m4, ml.c4, and ml.c5 instance families for training

> What is the best way to split time series data when using Amazon Machine Learning?

Allow Amazon Machine Learning to split sequentially

A simple way to split your input data for training and evaluation is to select non-overlapping subsets of your data while preserving the order of the data records. This approach is useful if you want to evaluate your ML models on data for a certain date or within a certain time range. For example, say that you have customer engagement data for the past five months, and you want to use this historical data to predict customer engagement in the next month. Using the beginning of the range for training, and the data from the end of the range for evaluation might produce a more accurate estimate of the model’s quality than using records data drawn from the entire data range.

> You are working with several scikit-learn libraries to preprocess and prepare your data. You also have created a script that trains your model using scikit-learn. You have been tasked with using SageMaker to train your model using this custom code. What can be done to run scikit-learn jobs directly in Amazon SageMaker?

Include your training script within a Notebook instance on Amazon SageMaker. Construct a sagemaker.sklearn.estimator.sklearn estimator. Train the model using the pre-build container provided by the Estimator.

You can run and package scikit-learn jobs into containers directly in Amazon SageMaker.

See https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-mleap-scikit-learn-containers.html

> You are working for an online shopping platform that records actions made by its users. This information is captured in multiple JSON files stored in S3. You have been tasked with moving this data into Amazon Redshift database tables as part of a data lake migration process. Which of the following needs to occur to achieve this in the most efficient way?

* Use COPY commands to load the tables from the data files on Amazon S3.
* Launch an Amazon Redshift cluster and create database tables.
* Troubleshoot load errors and modify your COPY commands to correct the errors.

You can add data to your Amazon Redshift tables either by using an INSERT command or by using a COPY command. At the scale and speed of an Amazon Redshift data warehouse, the COPY command is many times faster and more efficient than INSERT commands. You can load data from an Amazon DynamoDB table, or from files on Amazon S3, Amazon EMR, or any remote host through a Secure Shell (SSH) connection. When loading data from S3, you can load table data from a single file, or you can split the data for each table into multiple files. The COPY command can load data from multiple files in parallel

See [here](https://docs.aws.amazon.com/redshift/latest/dg/t_Loading_tables_with_the_COPY_command.html)

> You are in charge of training a deep learning (DL) model at scale using massively large datasets. These datasets are too large to load into memory on your Notebook instances. What are some best practices to use to solve this problem and still have fast training times?

* Pack the data in parallel, distributed across multiple machines and split the data into a small number of files with a uniform number of partitions.
* Once the data is split into a small number of files and partitioned, the preparation job can be parallelized and thus run faster.

When you perform deep learning (DL) at scale, for example, datasets are commonly too large to fit into memory and therefore require pre-processing steps to partition the datasets. In general, a best practice is to pack the data in parallel, distributed across multiple machines. You should do this in a single run, and split the data into a small number of files with a uniform number of partitions. When the data is partitioned, it is readily accessible and easily fed in as batches across multiple machines. When the data is split into a small number of files, the preparation job can be parallelized and thus run faster. You can do all of this using frameworks such as MapReduce and Apache Spark. Running an Apache Spark cluster on Amazon EMR provides a managed framework that can process massive quantities of data.

See [here](https://d1.awsstatic.com/whitepapers/aws-power-ml-at-scale.pdf)

> Type I and Type II errors

See [here](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)

> A machine learning specialist is running a training job on a single EC2 instance using their own Tensorflow code on a Deep Learning AMI. The specialist wants to run distributed training and inference using SageMaker. What should the machine learning specialist do?

* Use Tensorflow in SageMaker and edit your code to run using the SageMaker Python SDK

When using custom TensorFlow code, the Amazon SageMaker Python SDK supports script mode training scripts. Script mode has the following advantages: Script mode training scripts are more similar to training scripts you write for TensorFlow in general, so it is easier to modify your existing TensorFlow training scripts to work with Amazon SageMaker. Script mode supports both Python 2.7- and Python 3.6-compatible source files. Script mode supports Horovod for distributed training.

See [here](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html)

> You're a machine learning specialist working for an automobile broker who is looking to use machine learning to determine different models of muscle cars. You have been tasked with preparing a machine learning model to classify the different models of cars. The current implementation is using a neural network to classify other objects. What changes can be applied to help classify different models of muscle cars?

* Keep initial weights and remove the last layer

When you’re re-purposing a pre-trained model for your own needs, you start by removing the original classifier or last layer, then you add a new classifier that fits your purposes, and finally you can train the entire model on a new large dataset. 


From "Exam Readiness: AWS Certified Machine Learning - Specialty"

> A financial planning company is using the Amazon SageMaker endpoint with an Auto Scaling policy to serve its forecasting model to the company’s customers to help them plan for retirement. The team wants to update the endpoint with its latest forecasting model, which has been trained using Amazon SageMaker training jobs. The team wants to do this without any downtime and with minimal change to the code. What steps should the team take to update this endpoint?

Deregister the endpoint as a scalable target. Update the endpoint config using a new endpoint configuration with the latest model amazon s3 path. Finally, register the endpoint as a scalable target again.

> A real estate startup wants to use ML to predict the value of homes in various cities. To do so, the startup’s data science team is joining real estate price data with other variables such as weather, demographic, and standard of living data.
>
> However, the team is having problems with slow model convergence. Additionally, the model includes large weights for some features, which is causing degradation in model performance.
> 
> What kind of data preprocessing technique should the team use to more effectively prepare this data?

Standard scaler

> A Data Scientist wants to use the Amazon SageMaker hyperparameter tuning job to automatically tune a random forest model.
>
> What API does the Amazon SageMaker SDK use to create and interact with the Amazon SageMaker hyperparameter tuning jobs?

HyperparameterTuner()

"Whiz labs"

Review this - https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/

Support vector machines

https://scikit-learn.org/stable/modules/svm.html

Transcribe - streaming transcription - https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html

SageMaker Processing - see https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html

Bring your own sage maker kernel - https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi.html

Drop out - https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

Under and over fitting - https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

## ML Blog - Select Articles

https://aws.amazon.com/blogs/machine-learning/page/30/

https://aws.amazon.com/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/

https://aws.amazon.com/blogs/machine-learning/simplify-machine-learning-inference-on-kubernetes-with-amazon-sagemaker-operators/

https://aws.amazon.com/blogs/machine-learning/flagging-suspicious-healthcare-claims-with-amazon-sagemaker/

https://aws.amazon.com/blogs/machine-learning/making-accurate-energy-consumption-predictions-with-amazon-forecast/

https://aws.amazon.com/blogs/machine-learning/reduce-ml-inference-costs-on-amazon-sagemaker-for-pytorch-models-using-amazon-elastic-inference/

https://aws.amazon.com/blogs/machine-learning/pruning-machine-learning-models-with-amazon-sagemaker-debugger-and-amazon-sagemaker-experiments/

https://aws.amazon.com/blogs/machine-learning/increasing-performance-and-reducing-the-cost-of-mxnet-inference-using-amazon-sagemaker-neo-and-amazon-elastic-inference/

https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/

https://aws.amazon.com/blogs/machine-learning/using-amazon-textract-with-amazon-augmented-ai-for-processing-critical-documents/

https://aws.amazon.com/blogs/machine-learning/catching-fraud-faster-by-building-a-proof-of-concept-in-amazon-fraud-detector/

https://aws.amazon.com/blogs/machine-learning/ml-explainability-with-amazon-sagemaker-debugger/


https://aws.amazon.com/blogs/machine-learning/creating-a-complete-tensorflow-2-workflow-in-amazon-sagemaker/

https://aws.amazon.com/blogs/machine-learning/omnichannel-personalization-with-amazon-personalize/


https://aws.amazon.com/blogs/machine-learning/managing-missing-values-in-your-target-and-related-datasets-with-automated-imputation-support-in-amazon-forecast/

https://aws.amazon.com/blogs/machine-learning/a-b-testing-ml-models-in-production-using-amazon-sagemaker/

https://aws.amazon.com/blogs/machine-learning/scheduling-jupyter-notebooks-on-sagemaker-ephemeral-instances/

https://aws.amazon.com/blogs/machine-learning/optimizing-i-o-for-gpu-performance-tuning-of-deep-learning-training-in-amazon-sagemaker/

https://aws.amazon.com/blogs/machine-learning/deploying-your-own-data-processing-code-in-an-amazon-sagemaker-autopilot-inference-pipeline/


https://aws.amazon.com/blogs/machine-learning/the-importance-of-hyperparameter-tuning-for-scaling-deep-learning-training-to-multiple-gpus/

https://aws.amazon.com/blogs/machine-learning/horovod-mxnet-distributed-training/

https://aws.amazon.com/blogs/machine-learning/building-a-customized-recommender-system-in-amazon-sagemaker/


https://aws.amazon.com/blogs/machine-learning/automated-monitoring-of-your-machine-learning-models-with-amazon-sagemaker-model-monitor-and-sending-predictions-to-human-review-workflows-using-amazon-a2i/

https://aws.amazon.com/blogs/machine-learning/streamline-modeling-with-amazon-sagemaker-studio-and-amazon-experiments-sdk/

Start...

https://aws.amazon.com/blogs/machine-learning/page/10/



### Other Notes

On Sampling

* https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis
* https://medium.com/@hazy_ai/imbalanced-data-and-credit-card-fraud-ad1c1ed011ea

Study Next

* SageMaker
* Glue ETL
* Glue & Spark Integration