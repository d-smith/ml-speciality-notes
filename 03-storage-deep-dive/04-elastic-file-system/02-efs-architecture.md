# Amazon EFS Architecture

## Deployment Models

* On premises NFS storage
    * Complex - capacity, performance management, expected data growth, design for HA
    * Physical location available where you need your servers?
* Cloud-based DIY
    * EC2 instances, monitoring and patching, EBS volume monitoring, maintenance, patching
    * Do for each AZ
    * Routes and availability config
    * Smart client routing/recovery for HA
* EFS
    * Just create mount targets in the AZs where your ec2 instance are
    * NFS 4.0 and 4.1
    * Pay only for what you store, optionally you can pay for provisioned throughput as well

* Resources
    * EFS file system - strong data consistency and file locking
        * Regional construct
        * 125 file systems per account
        * Two performance modes - general purpose and max i/o
        * Two throughput modes - burst and provisioned
        * Accessible from Amazon EC2
        * Accessible from on-premises using direct connect
        * Control access using POSIX permissions
    * Tags - use to organize your file systems
    * Mount target - resource created to enable access to an EFS from EC2 or on-prem
        * One per AZ, like an NFS mount point
        * Created automatically, has IP address and DNS name
        * Can pick the subnet in an AZ in you have multiple subnets in an AZ, mount target gets IP from the subnet - can be static or dynamic addess. IP addresses do not change.

* Requirements
    * You can mount targets to instances in one VPC at a time
    * Both EFS file systems and VPC must exist in the same region

* Access in 3 ways
    * EC2 instances
    * Corp data center via direct connect
    * From hosts running in the vmware cloud on AWS

* Direct connect
    * Mount through mount targets, AWS recommends using IP address not DNS name to avoid integrating DNS services with on-premises DNS domains
    * Use cases: migration, backup/dr, bursting. Bursting using copying on-permieses data to EFS, analyzing the data at high speed, and sending it back to the on-premises servers.

* Amazon EFS File Sync
    * Connect to data in existing on-prem or in-cloud file systems
    * Copy data 5x faster than standard linux copy tools
    * Set up and manage from the AWS console



