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