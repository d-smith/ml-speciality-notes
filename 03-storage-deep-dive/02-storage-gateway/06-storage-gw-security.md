# Storage Gateway Security

* Access to AWS Storage Gateway requires credentials that AWS can use to authenticate your requests

Identities

* AWS account user - don't use it, use it to create your first IAM user
* IAM user
* IAM role

What is CHAP?

* Authentication between iSCSI targets and initiators via Challenge-Handshake Authentication Protocol (CHAP)
    * Protects against playback and man in the middle attacks
    * Configure both in the storage gateway console and the iSCSI initiator software you use to connect to the target
    * Can have one or more credentials per target

Access Control

* Requires both credentials to authenticate your requests, plus permissions to perform tasks
* File share - file gateway requires access to upload files into you S3 bucket, associated policy enabled the permissions

Resources

* Primary resource: Storage gateway 
* Sub-resources:
    * File share 
    * Virtual tape library
    * Cached volume, non-cached volume
    * iSCSI target
* Resource owner - the AWS account that created the resource

End to End Security

* CHAP - secures iSCSI interface
* Direct connect - can keep the traffic off the public internet
* TLS protection of data transferred to AWS, SSE used to protect data at rest in AWS