# Implementation and Operations

## Introduction

Machine learning cycle - covers deploy to production, monitor and evaluate post deployment

## Concepts

| | Offline Usage | Online Usage |
| -- | -- | -- |
| What | Makes inferences on datasets in batch and returns results as a set | Makes inferences on demand as the model is called and returns results immediately |
| Why | Entire dataset is needed for inferences; pre-process data before using as an input for another model | Need instant response when endpoint is called via app or service |
| When | - Predictive models with large historic dataset inputs<br> -Feature engineering for a follow on model | - Real-time fraude detection<br>- Autonomous machines |

Types of Deployments

| | Big Bang | Phased Rollout | Parallel Adoption |
| -- | -- | -- | -- |
| Time | Least amout of time | More time | Most amount of time |
| Risk | High | Low | Low |
| Cost | f(Risk, Time) | f(Risk, Time) | f(Risk, Time) |

Sometime risk is amplified in parallel adoption due to concurrency such as synchronization issues, multiple processes, temporary integrations, etc.

Rolling Deployment

* Rather than upgrade all the resources at once, the upgrade is done one by one to minimize downtime.

Canary Deployments

* Deploy a new version into production to see how it works, route small amount of traffic to it.
* Use route 53 to distribute traffic

A/B Testing

* Deploy a new version into production and configure a set amount of new inbound traffic to use the new (B) version, recording follow-on data about the outcome of those who used the new version. Once enough data is collected, make a decision on whether to fully deploy the new version or make adjustments.
* SageMaker hosting can do this

CI, CD, and More

* Continuous Integration - merge code changes back to main branch as frequently as possible with automated testing as you go.
* Continuous Delivery - you have automated your release process to the point you can deploy at the click of a button.
* Continuous Deployment - each code change that passes all stages of the release process is released into production with no human intervention required.

## AI Developer Services

AWS AI Stack

* AI Services - for app developers, no ML experience required
    * Amazon Comprehend
    * Amazon CodeGuru
    * Amazon Lex
    * Amazon Forecast
    * Amazon Polly
    * Amazon Rekognition
    * Amazon Textract
    * Amazon Transcribe
    * Amazon Translate
    * Amazon Kendra
    * Amazon Personalize
    * Amazon Fraud Detector
* ML Services - ML developers and data scientists
    * SageMaker
        * Ground Truth
        * Training
        * Notebooks
        * Hosting
        * Algorithms
        * Marketplace
* ML Frameworks and Infrstructure - ML researchers and academics
    * Frameworks - mxnet, tensortflow
    * Interfaces - gluon, keras
    * Amazon Greengrass
    * Amazon EC2
    * AWS Deep Learning AMIs

AWS Developer Services

* Easy to use with no ML knowledge required
* Scalable and robust
* Redundant and fault tolerant
* Pay per use
* REST API and SDK

Amazon Comprehend

* Natural language processing (NLP) service that finds insights and relationships within text
* Use case example: sentiment analysis of social media

Amazon Forcast

* Combines time-series data with other variables to delivery highly accurate forecasts
* Example usage: forecast seasonal demand for a specific color of shirt

Amazon Lex

* Build conversational interfaces that can understand the intent and context for natural speech
* Example use: create a customer service chatbot to automatically handle routine requests


Amazon Personalize

* Recommendation engine as a service based on demographic and behavioral data.
* Example use: Provide potential upsell products at checkout during a web transaction.

Amazon Polly

* Text-to-Speech service supporting
multiple languages, accents and
voices.
* Example use: Provide dynamically generated
personalized voice response for
inbound callers.

Amazon Rekognition

* Image and video analysis to parse and
recognize objects, people, activities
and facial expressions.
* Example use: Provide an additional form of employee
authentication though facial recognition
as they scan an access badge.

Amazon Textract

* Extract text, context and metadata
from scanned documents
* Example use: Automatically digitize and process
physical paper forms

Amazon Transcribe

* Speech-to-Text as a service
* Example use: Automatically create transcripts of
recorded presentations.

Amazon Translate

* Translate text to and from many
different languages
* Example use: Dynamically create localized web
content for users based on their
geography.

Amazon CodeGuru

* Amazon CodeGuru is a developer tool powered by machine learning that provides intelligent recommendations for improving code quality and identifying an application’s most expensive lines of code
* Amazon CodeGuru Profiler helps developers find an application’s most expensive lines of code along with specific visualizations and recommendations on how to improve code to save money.
* Amazon CodeGuru Reviewer uses machine learning to identify critical issues and hard-to-find bugs during application development to improve code quality.

Amazon Kendra

* Amazon Kendra is a highly accurate and easy to use enterprise search service that’s powered by machine learning. Kendra delivers powerful natural language search capabilities to your websites and applications so your end users can more easily find the information they need within the vast amount of content spread across your company.

Amazon Fraud Detector

* Amazon Fraud Detector is a fully managed service that uses machine learning (ML) and more than 20 years of fraud detection expertise from Amazon, to identify potentially fraudulent activity so customers can catch more online fraud faster.