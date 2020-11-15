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



Data Set Analysis

Coumpter Vision