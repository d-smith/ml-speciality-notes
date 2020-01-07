# Data LifeCycle Management

Concept: use a single object management API (the S3 API) and use the lifecycle to 
transition objects into glacier when appropriate.

Lifecycle Policies

* Rules are composed of transition and expiration actions
* Move objects into different storage classes (s3 standard, standard ia, onezone ia, glacier)
* Objects moves on policy conditions

Classifying Workloads

* Hot data - active or temporary s3 standard
* Warm data - infrequently accessed - standard ia
    * Onezone IA - save some money if you can handle the reduced availability
* Cold data - archive and compliant data - glacier

Each class has the same durability

Standard IA

* Must stay for at least 30 days - delete it sooner, charged for 30 days
* Before transitioning data to S3 standard IA, ask how frequently is the data accessed, and how long will it be stored in IA

Glacier

* not suitable if ms response times needed for retrieve
* 90 day minimum for object storage
* 3 retrieval options
    * Standard - 3 to 5 hours
    * Expidited - 1 to 5 minutes
    * Bulk - 5 to 12 hours

Storage Class Analysis

* S3 feature
* Monitors access patterns to understand your storage usage
* After 30 days, recommends when to move objects to s3 infrequent access
* Export file includes daily report of storage, retrieved bytes, and gets by object age

Differentiating S3 and Glacier

* S3 - reads are synchronous
* Glacier - reads are asynchronous: first call restore, then get object after restore is complete

Life Cycle policy

* Consists of a set of rules with pre-defined actions
* Each rule consists of...
    * A filter specifying the subset of objects it applies to. Object are identified by tag,
    prefix, or a combination of the two.
    * A status indicating if the rule is in effect
    * One or more lifecycle transitions or expirations

Example:

```xml
<LifecycleConfiguration>
    <Rule>
        <Filter>
            <Prefix>audit</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Transition>
            <Days>365</Days>
        </Transition>
        <StorageClass>Glacier</StorageClass>
    </Rule>
</LifecycleConfiguration>
```

Apply to all objects? Leave the Prefix element blank

Zero Day lifecycle policies

* Specify a life cycle policy using tags to select individual or sets of objects, with a transition with empty Days tags to move them directly into Glacier




