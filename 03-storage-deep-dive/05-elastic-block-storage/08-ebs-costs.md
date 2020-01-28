# Tracking EBS Usage and Costs

EBS pricing includes:

* Volume storage - GB provisioned per month
* Snapshot storage - amount of space your data consumers in Amazon s3
* Throughput for provisioned IOPS volumes

Snapshot Copy Costs

* Charge for the data copied across regions plus standard snapshot costs

Use Tags to Assign Key/Value pairs

* User defined tags
    * Use for identifying and managing snapshots
    * Active as [cost allocation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/cost-alloc-tags.html) tags.