# Managing Storage in Amazon S3

## Bucket Options

Storage management tools

* life cycle policies, object tags, s3 inventory, cross region replication, cloudwatch metrics, event notifications, storage class analysis, cloudtrail events

Bucket Options

### Versioning

* Unversioned, versioning enabled, versioning suspended
* Applied at the bucket level
* Can leverage for avoiding accidental deletes and overwrites

Version-enabled delete

* Creates a delete marker, can delete the delete marker to restore the object

Versioning Suspended

* You cannot turn it off and remove all versions of objects
* Only effects future operations on objects in the bucket. When you suspend versioning, nothing is changed on the exiting objects.
* Adds a null version id to every subsequent put, post, or copy to any new or existing object in the bucket
* This means any object with a null version id will be overwritten. Objects with a current version id of other than null will not be overwritten,  but rather, a new object with a null version id will be created.



### Server access logging

Capture access request details

* requestor, bucket name, request time, request action, response status, error code 
* To enable:
    * Turn on log delivery
    * Grant s3 log delivery group write permission on the bucket where you want the logs stored

### Object level logging

* Record object-level API activity using the CloudTrail data events feature (additional cost) - who, when, what resources
* Monitor changes to bucket configurations
* Log object level operations
* CloudWatch event and log aggregation


### Static website hosting

* Host a static website using only Amazon S3 without any additional servers.
* Access at bucket-name.s3-webside-AWS REGION.amazonaws.com

### Default encryption

* Can ensure objects are encrypted by default

### Tags

* Manage data based on the nature of your data, and apply more granular controls
* Tags are key-value pair, up to 10 tags per object
* Use with access control and lifecycle policies
* Use with s3 analytics, display information by tag in Amazon cloud watch 

Example: only allow access to objects containing PII to HR group

* Tag with pii true
* Use a bucket policy like

```json
{
    "Statement":[
        {
            "Effect":"Allow",
            "Principal":"arn:aws:iam:111122223333:group/HR",
            "Action":[
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::hrbucket/*"
            ],
            "Condition": {
                "StringEquals": {
                    "s3:ExistingObjectTag/pii":"true"
                }
            }
        }
    ]
}
```

Other condition example:

```json
"Condition":{
    "StringEquals":{
        "s3:RequestObjectTag/Project":"X"
    }
}
```

### Transfer Acceleration

* Increase transfer speeds over long distances between your client and s3
* Enabling provides a new URL to use with your application.

### Event Notifications

* Notify when objects are create via put, put, copy, delete, or multipart uploads
* Filter on prefixes and suffixes
* Trigger workflow with Amazon SNS, AMazon SQS, and Amazon lambda functions

Benefits:

* Simplicity
* Speed
* Integration

### Requestor Pays

* The requestor pays the cost of the request and the data transfer

## Object Lifecycle Management

Lifecycle configuration

* Enables you to specify the lifecycle management of objects in a bucket
* Contains a set of one or more rules
* Exach rule defines an action:
    * Transition
    * Expiration

Automate Transitions

* Automate the tiering process from one storage class to another
* Considerations:
    * No automatic transition of objects less than 128KB in size to to standard IA or one zone IA
    * Data must remain on its current storage clas for at least 30 days before it can be automatically moved to S3 standard IA or One Zone IA
    * Data can be moved from any storage class directly to Amazon Glacier

Action Types

* You can direct S3 to preform specific actions in an object's lifetime by specifying one or more of the following predefined action in a lifecycle rule. The effect of these action depends on the versioning state of your bucket.
    1. Transition: moves objects to standard IA, onezone IA, or glacier based on the object age you specify
    2. Expiration: deletes objects after the time you specify

For versioning-enabled buckets there are additional action elements:

* NoncurrentVersionTransition
* NoncurrentVersionExpiration

Transitioning Objects

* From S3 standard, can transition to S3 IA, S3 one zone IA, Glacier.  Objects must be > 128KB, must be stored > 30 days
* From S3 IA, can transition to  S3 one zone IA, Glacier. Objects must be stored for at least 30 days.
* From S3 one zone IA, can transition to Glacier.
* Cannot transition from Glacier.
* Objects transitioned to Glacier are still s3 objects, and are restored and accessed via the S3 console and APIs.