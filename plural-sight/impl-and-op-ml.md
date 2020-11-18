# Implementing and Operating AWS Machine Learning Solutions

## AWS Machine Learning Services

THe exam

* Data engineering
* Exploratory data analysis
* Modeling
* Machine Learning Implementation and Operations

### Data Ingestion and Transformation Services

Ingestion

* Kinesis Data Streams
    * Manual configuration for scale
    * Real time
* Kinesis Data Firehose
    * Fully managed
    * Near real time
* Kinesis Video Stream
    * Fully managed
* Kinesis Data Analytics
    * Enables queries to be run against data stream

Data Transformation

* Amazon EMR
    * AWS big data platform
    * Frameworks - spark, hive, hbase, flink, hudi, presto
* AWS Glue
    * Fully managed ETL service
    * Includes glue crawler
    * Data source: s3, dynamodb, jdbc (rds, redshift)
* Kinesis Data Firehose Data Transform
    * Invocation time is 5 minutes
    * Payload limit (in and out) is 6MB

### AWS AI Services

Suite of service that enable developers to quickly integrate deep learning into their applications through pre-trained models that are targeted at specific use cases.

Language Analysis

* Amazon Comprehend - NLP to extract insights from text inputs
    * Sentiment analysis on unstructured text
    * Categorize documents by topics
    * Medical sub-service
* Lex - conversational experinces
    * Extracts intent from text or audio
    * Chatbot for customer service
    * Integrate with call center for automated data retrieval
* Amazon Polly - lifelike audio of speech from text
    * Automated audio on a call for dynamic information
    * Enable accessibility of content
    * Lifelike audio in embedded devices
* Amazon Textract = extract insights and data from scanned documents
    * AUtomating the importing of documents into an application
    * Developing a classification workflow for documents
* Amazon Translate - uses nueral machine translation (NMT) to translate input input text
    * Supports 55 languages
    * Automate localization process for a digital experience
    * Enable cross language search
* Amazon Transcribe
    * Speech to text from audio
        * Supports batch and real-time processing of audio
        * Automate the generation of captions for video content
        * Include medical sub-service

Data Set Analysis AI Services

* Amazon Forecast
    * Forcasting on historical time series data
    * Supply chain needs based on projected manfacturing deman
    * Predict sales data based on multiple historical data sets
* Amazon Fraud Detector
    * Identify fraudulent payment transactions
    * Identify fraudulent acccount creation
    * Identify fraudulent actions taken on a legitimate account
* Amazone Personalize
    * Creates a personalization engine for users from user behavior data
    * Personalized content or product recc
    * Enabling personalized promotions

Computer Vision AI Services

* Amazon Rekognition
    * Automated analysis of image and video data
        * Text extraction, object recognition, face/emotion detection
        * Enabling search across a large library of digital content
        * Implement a facial authentication system
        * Automating image moderation



### Amazon SageMaker Services


* SageMaker ground truth
    * Use a provided workforce or workforce via mechanical turk
    * Enable automated labeling too
* SageMaker Notebooks
    * Managed computer environments for using Jupyter notebooks
    * Integrates with storage volumes that can persist between runs
* SageMaker Studio
    * Integrated dev environment for machine learning in the browser
    * Provides support for building, tuning, deploying, and managing models
    * Built on JupyterLab and deeply integrated with SageMaker
* SageMaker Autopilot
    * Automates building, training and tunint of machine learning models
    * Automate model creation while allowing customization
* SageMaker Automatic Model Tuning
    * Optimize ML hyperparameters in your model training
    * categorical, continuous, and integer types
* SageMaker Experiments
    * Enables organization of training jobs and related analtics at scale
* SageMaker Neo
    * Model optimization and cross-platform compilation
* Amazon Augmented AI (A2I)
    * Integrates human review into inference process
    * Provides a configurable workflow for situation when accuracy is critical
    * Integrated into textract and rekognition

Amazon EC2

* Can work on EC2 directly with dep learning AMI

Container Services

* AWS provides deep learning containers preconfigured for specific framework

Amazon EMR

* Support for Jupyter notebooks, ML frameworks

On-premise Server

* Use pre-built SageMaker Docker images on prem
* Can train on prem, run in SageMaker for inference, and vice-versa

## Model Deployment

Deployment Approaches in SageMaker

* Hosting Services - Deploy an interence endpoint to integrate inference into your workflow or application
* Batch Transform - perform an inference job on an entire dataset stored in Amazon S3

Hosting Services Use Cases

* Deploy inference to a secure API endpoint
* Integrate inference into an application or workflow
* Don't have all the data you'll need in the future

Batch Transform Use Cases

* Do not need an exposed endpoint for inference
* Already have the entire dataset on which you want to perform inference
* Need to process an incoming dataset to remove records prior to inference

### SageMaker Batch Transform

Utilizing Batch Transform

* Post training validation of model fit
* Chain multiple models
* Predictions that will be served outisde of SageMaker

Batch Transform Process

* Training and store model
* Load dataset for inference into Amazon S3
* Specify configuration for batch transform job
* Execute job on dataset
* Results are stored in specified s3 bucket

Batch Transform Config

* Input dataset -\> inference infrastructure -\> output data
* Infrastructure can be configured with the number of instances and containers
* Strategy dictates how data is pulled from the input source (batched or record at a time)
* Transform data source input defines the source s3 bucket
* The AssembleWith parameter dictates how the data will be output
* The transform output data location specifies the output s3 bucket 

Infrastructure Approaches

* SageMaker Container
* Customer container 

### Sage Maker Hosting Services

Provide a scalable real-time inference REST endpoint for your machine learning models that can be integrated into your applications and workflows

Example:

* Client app -\> API gateway -\> Lambda -\> Endpoint -\> Load balancer -\> Instances -\> model in s3

Deploying to SageMaker

* Create a model - name, s3 location
* Endpoint configuration - model, instance size, no initial instances
* Create an endpoint from the model configuration

Inference Endpoints

* Secured with HTTPS
* Can autoscale based on defined minimum and maximums
* Can be updated without downtime
* Support A/B testing through multiple production variants

Multiple Production Variants

* Variants defined are defined in the endpoint configuration
* Every endpoint configuration has at least on production variant
* Multiple production variants allow you to A/B test multiple versions of a model
* The InitialVariantWeight parameter determines the traffic to the variant
* Weights per variant can be updated by updating the endpoint configuration.

Deploying a Model using SageMaker Hosting Services

* Provided notebook, python 3 (data science) notbok type
* Train the model
    * XGBoost container
    * Create estimator, set hyperparameters, train it
* Create the model
    * Ref the model data from training
* Create an endpoint configuration
    * Dictates how inference will work on that endpoint
* Create an endpoint
    * Using the endpoint configuration

Validate Model and Deployment

* Loop over some data and invoke the endpoint, examine the output

Update with Multiple Production Variants

* Retrain a model, maybe adjust the hyperparameters
* Save the model
* Create a new endpoint configuration with multiple variants
* Update endpoint with new endpoint configuration
* Test it out, look at InvokedProductionVariant in the response

Additional Deployment Topics

* Amazon Elastic Inference
    * Access to a factional GPU at a cost that is greatly reduced over a full GPU allocation for frameworks where Elastic Ingerence capability is supported
* Multi-model Endpoints
    * Different from multple variants of the same
    * Endpoints can serve multiple distinct models
    * Supports multiple production variants on multiple models
    * Does not support serial inference pipelines
    * Does not support GPU instance type or elastic inference
    * Share instances among multiple models
* Inference Pipelines
    * An inference pipeline is an Amazon SageMaker model that is composed of a linear sequence of 2 to 5 containers that process requests for inferences on data
    * HTTP between the container
    * Containers located on the same EC2 instances
    * Enables a complete workflow without having to do external data preprocessing or postprocessing

