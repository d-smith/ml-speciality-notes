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




# Glacier Durability and Security

Durability

* Stored in 3 AZs per region
* 11 nines of durability
* Each region has two redundant transit centers
* Highly peered and connected facilities
* Metro fiber for intra-AZ connections, inter, and transit center connections

Security

* Resources stored in Glacier are protected by the vault
* Only vault owners have access to resources created created on Glacier
* Control to resources controlled by IA policies and value access policies
* Automatically encrypts data at rest, and protects data in transit via TLS
* Supports S3 SSE
    * If SSE enabled on S3 before uploading objects, encryption is applied to objects in s3, objects transitioned to Glacier, and objects restored from Glacier to Amazon S3.
* Audit support via CloudTrail, audit logs stored in S3
    * Log details - requests to glacier, source ip of requests, time of request, originator of request
    * SNS alerts when new logs available

Vault Lock

* Vault lock policy specifies a retention requirement in days from archive creation (not data of vault lock policy)
* Vault is locked against changes or deletion until the retention time has passed, acting as WORM storage
* Supports legal hold, which extends WORM for archives subject to a legal event
* Locking a policy means it cannot be changed, and Amazon will execute the policy
    * Vault lock means customers do not need to purchase WORM storage
    * Enforces 7 year retention policies for example
* One vault lock policy for vault, vault access policies are different. Both can be used together.
* Two step locking policy
    * Attach a vault lock policy
    * Complete the lock using the lock id associated with the vault lock attachment before it expires in 24 hours
    * InitiateVaultLock - activates testing retention policy, returns unique lock id which expires in 24 hours
    * AbortVaultLock - deletes in process policy, modify a policy before locking it down
    * CompleteVaultLock - lock down vaule with lock ID, vault lock cannot be aborted

Vault Tags

* Vaults can be tagged with up to 50 tags
* Applied to the vault resource
* Can use legal hold tags in vault lock policies for example to Deny a delete:


```json
"Condition":{
    "NumericLessThan":{
        "glacier:ArchiveAgeInDays":"365"
    },
    "StringLike":{
        "glacier:ResourceTag/LegalHold"[
            "true",
            ""
        ]
    }
}
```

Cohasset Associates Certification

* SEC 17a-4(f) and CFTC 1.31(b)-(c) compliance assessment - financial services books and records

