# Using ML Models in Production

Productizing a ML Model

* Integrating an ML model with existing software
* Keeping it running successfully over time

Aspects to Consider:

* Model hosting
* Model deployment
* Pipelines to provide feature vectors
* Code to provide low-latency and/or high volume predictions
* Model and data updating and versioning
* Quality monitoring and alarming
* Data and model security and encryption
* Customer privacy, fairness, and trust
* Data provider contractual constraints (e.g. attribution, cross-fertilization)

Types of production environments

* Batch predictions
    * Useful if all possible inputs known a priori (e.g. all product categories for which demand is to be forecase, all keywords to bid)
    * Predictions can still be served real-time, simply read from pre-computed values
* Online predictions
    * Useful if input space is large (e.g. customer's utterances or photos, detail pages to be translated)
    * Low latency requirements (e.g. at most 100ms)
* Online training
    * SOmetimes training data patterns change often, so need to train online (e.g. fraud detection)