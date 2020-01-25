# Introduction

Block data - requires operating system has direct byte-level access to storage device.

File storage - NFS protocols abstract the operating system from the storage devices.

Object - application access content through RESTful APIs

What is EBS?

* Block storage as a service
* Enables you to:
    * Create and attach volumes through APIs
    * Access the service over the network
* An EBS volume is not a single physical disk

EBS volume

* EBS is a distributed systen
* Each EBS volume is made up of multiple, physical devices
* Created in an availability zone, can be attached to any ec2 instance in that availability zone
* Can take a snapshot of the data, then create a new volume from that snapshot in other AZs if desired
* Volume data persists independently from the instance it is associated with, can be detached from one instance and attached to another, etc.
* Can be attached to one instance at a time, but many volumes can be attached to a single instance

HA and  Durability

* Each volume has 5 9s service availability 99.999%
* Annual failure rate is 0.1% to 0.2%

Block Storage Options

* EC2 instance store - ssd or hdd
* EBS SSD-backed volumes - gp2 or io1
* EBS HDD-backed volumes - st1 or sc1

Instance store 

* Temporary block-level storage for EC2 instance
* Disjs physically attached to the host computer
* Ideal for temporary storage - buffersm cachesm scratch data, replicated data, etc

EC2 Instance Store vs EBS

* Both volume types are:
    * Presented as block storage to EC2
    * Available as SSD or HDD
* EC2 instance store has:
    * Ephemeral (non persistent) data store
    * No replication (by default)
    * No snapshot support

Use Cases

* Relational database
* Enterprise applications
* Development and test
* NoSQL databases
* Business Continuity

