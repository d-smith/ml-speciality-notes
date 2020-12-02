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


