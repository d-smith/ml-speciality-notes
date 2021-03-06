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

## Securing SageMaker

Shared Responsibility Model

* AWS - security of the cloud
* Customer - security in the cloud

Customer Concerns in ML

* Data Protection
* AuthN abd AuthZ
* Compliance
* Monitoring
* Securing Infrastructure

### Securing Data

Aspects of data encryption

* Data at rest
* Date in transit

S3 Data

* Enable encriptions on s3
    * Bucket policy to enfore
    * Use customer managed key
* SageMaker verifies that its data outputs are encrypted

SageMaker Data Volumes

* Notebook Instances
* Processing Jobs
* Training Jobs
* Batch transform jobs
* Hosting services endpoints
* Hyperparameter tuning jobs


Data in Transit

* Most inter-network traffic data is encrypted using TLS 1.2
* Some areas are not encrypted
    * Commumication between control and training job instances
    * Node communication in distributed processing jobs
    * Node communication in distributed training jobs

Amazon Macie

* Fully managed service focused on data security
* Utilizes machine learning to discover and report on data stored in S3
* Alerts organizations on several conditions:
    * Unusual data access patterns
    * Configuration errors for sensitive data
* Enables automated compliance checking for data stored in s3

### VPC config for SageMaker

When you use SageMaker in VPC, you can take advantage of VPC endpoints to avoid sending traffic over the public internet.

Two Types of VPC Endpoints

* Redundant and highly-available
* Can control access using IAM policies
* Gateway endpoints - S3 and DynamoDB
* Interface endpoints
    * Leverage PrivateLink

### IAM for SageMaker

Using IAM with SageMaker

* Define least privilege access for each user using IAM policies
* Users should leverage MFA
* Supports identiry-based policies not resource based policies
* Start with the sample policies

### Securing Notebooks

* Notebook instances intended for a single users
* SageMaker notebooks are internet-enabled by default
    * Can be eliminated by launching notebook in a custoer managed vpc
* Can manage internet access when deployed to VPC using the usual controls

### Compliance

* Extract compliance report using AWS Artifact

## Implementing an HA Machine Learning Solution

Reliability on AWS

* Fault tolerance - being able to support the failure of components in your architecture
* High availability - keeping your entire solution running in the expected manner despite issues that may occur

Scaling SageMaker Endpoints

* Elasticity - aquire resources as you need them, release when not needed
* Vertical scaling - select larger instance types with additional resources
* Horizontal scaling - scale out and add additional resources as you need them

Instance Types

* Standard instance
* Memory optimized
* Compute optimized
* Accelerated computing
* Elastic inference

SageMaker Autoscaling

* Horizontal scaling approach
* Configure in the console, cli, or via api
* Uses cloud watch metrics to define the scaling poloicy
* Initial metric to utilize for scaling - SageMakerVariantInvocationsPerInstance, average times per minute the instance is invoked for a variant
* Can utilize other metrics, e.g. CPUUtilization

Configuring Autoscaling

* Select endpoint
* Select a production variant, configure autoscaling
    * Set max, min instances
    * Edit scaling policy - select metric, target value, cool down interval for scale out, scale in

### Deployment Methodologies

Continuous Integration

* Dev practive that requires developers to integrate code into a shared repo several times a day. Each checking is verified by an automated build, allowing teams to detect problems early.

Continuous Delivery

* Dev process where a solution is built and tested automatically based on a commit to a source repository without any human intervention. The final deployment process, though automated, is still triggered manually. 

Continuous Deployment

* Developmet process where a solution is built, tested, and deployed automatically based on a commit to a source repository without any human intervention.

### Monitoring SageMaker

* CloudWatch - metrics, logs
    * Metrics stored for 15 months, 2 weeks available in console
* CloudTrail - audit

### Fault Tolerant Endpoint Configuration

* Spread your damn instances across availability zones

### Loosely Coupling

By building a modular architecture, components can integrate with other components without having to know about specific definition or configuration of those components. Elements like load balancers, queues, and managed workflow engines support loose coupling.

* Step flow function
* SQS
* SNS
* Eventbridge