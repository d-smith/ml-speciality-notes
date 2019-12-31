# Using AWS Security Tools and Services with Amazon S3

## AWS Config

* Continuously monitors and records your aws  resource config
* Evaluate config against a set of configurations
* Notify you when out of compliance
* Keeps configuration history
* Config snapshots go in an s3 bucket, notifications via SNS topic

Build-in rules

* s3 bucket logging enable
* s3 bucket public read prohibited
* s3 public write prohibited
* s3 bucket ssl requests only
* s3 bucket versioning enabled

Can have custom rules too.

More info [here](https://docs.aws.amazon.com/config/latest/developerguide/WhatIsConfig.html)

## API Logging with AWS CloudTrail

* CloudTrail - perform security analysis, meeting your IT auditing and compliance needs, improve your security posture.
* Log bucket level operations (management events)
* Log object levels operations (data events)
* Also integrates with Cloud Watch, can take immediate action based on CloudWatch events

Permission are automatically attached when creating an Amazon S3 bucket

* As part of creating or updating a trail in the AWS CloudTrail console
* Using the AWS CLI create-subscription and update-subscription commands

If you specify an existing bucket as the destination, you must attach a [policy granting CloudTrail write permission](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/create-s3-bucket-policy-for-cloudtrail.html)

## Security Inspection

* Verify encryption status using Amazon S3 Inventory
* AWS trusted advisor will list buckets with open (public access). You can also set an alarm should any of the buckets fail the check. 
* The amazon s3 console also provides a public bucket access indicator.

## AWS Trusted Advisor

* Observe best practices
* Save money
* Improve performance
* Close security gaps

S3 checks

* AWS S3 bucket permissions
* AWS S3 bucket logging
* AWS S3 bucket versioning

Amazon Macie

* Security services that uses machine learning to look for sensitive data in your buckets and alerts you if you have unsecuried data.
* Can answer...
    * What data do i have in the cloud?
    * Where is it located?
    * How is data being shared and stored?
    * How can i classify data in near-real time?
    * What PII/PHI is possibly exposed?
    * How do i build workflow remediation for my security and compliance needs?

    