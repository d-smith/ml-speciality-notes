# Securing Your Data in Amazon EFS

Security Controls

* Network traffic - VPC security groups and network ACLs
* File and directory access - standard linux directory and file permissions
* Grant admin access using the API to file systems - AWS identity and access management
* Encrypt data at rest - KMS. Encryption can only be enabled during file system creation.
* Encrypt data in transit - Transport Layer Security

Security Groups

* Can configure up to 5 security groups per mount target
* SG to permit inbound traffic on port 2049. EC2 instances that are a member of that SG are allowed to mount the file system, others cannot.

AWS IAM

* IAM policies control who can use the administrative APIs