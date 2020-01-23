# Managing EFS File Systems

3 interfaces: management console, command line, SDKs

Create - choose region, choose a VPC and subnets, assign a security group, choose the performance mode (general purpose, max i/o), throughput mode (bursting, provisioned), choose to enable encryption.

Mount with NFS client or EFS mount helper.

Amazon EDS Utils - open source tools, including the EFS mount helper, encryption tools, etc.

```console
sudo yum install -y amazon-efs-utils

sudo mkdir efs

# without encryption in transit
sudo mount -f efs fs-ff44ff44:/ efs

# with encryption in transit
sudo mount -f efs -o tls fs-44ff44ff:/ efs
```

Before you mount, you must install the EFS client on the EC2 instance and create a firectory for the mount point.

Recommended mount config: I/O size parameters of 1MB, hard mount, timeout of 60 seconds (and no less than 15s), two minor timeouts and retranmissions before a major timeout

Can use fstab to automount on startup.

