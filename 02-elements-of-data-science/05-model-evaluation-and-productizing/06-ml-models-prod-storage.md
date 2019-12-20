# Using ML Models in Production: Storage

## Latency

* DynamoDB where sub 10ms latency needed, and want to scale. Good for online.
* S3 and Glacier - flexible storage, backup, transfer, archiving, etc. Good for offline.

## Data Storage Formats

* Row Oriented
    * Comma/tab separated (CSV,TSV)
    * Read-only DB (RODB): internal read-only file-based store with fast key-based access
    * Avro: allows schema evolution for Hadoop
* Column Oriented
    * Parquet: type-aware and indexed for hadoop
    * Optimized row columnar (ORC): type-aware, indexed, and with statistics for Hadoop
* User Defined Formats
    * JSON: for key-value objects
    * Hierarchical data format 5 (HDF5): flexible data model with chunks
* Compression can be applied to all formats
* Usual trade-offs: read/write speeds, size, platform-dependency, abilty for schema to evolve, schema/data separability, type richness

## Model and Pipeline Persistence

Predictive Model Markup Language (PMML)

* Vendor independent XML-based language for storing ML models
* Support varies in libraries
    * KNIME (analytics/ML library): full support
    * scikit-learnL extensive support
    * Spark MLlib: limited support

Custom methods

* scikit-learn: uses the python pickle method to serialize/deserialize Python objects
* Spark MLlib: transformers and estimators implement MLWritable
* TensorFlow: Allows saving of metagraph
* MxNet: Saves into JSON

## Model Deployment

Technology transfer: experimental framework might not suffice for production

* A/B testing or shadow testing: helps catch production issues early
* Zinkevich, Martin. [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)

## Information Security

* Make sure that you handle training and evaluation data in accordance with data classification
* Models may need to be trated with with the same classification level as the source data.