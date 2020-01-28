# EBS Performance and Monitoring

Factors that effect performance

* I/O characteristics of your applications
* Type and configuration of your ec2 instances
* Type and configuration of your volumes

Performance Tips

* Understand how performance is calculated
    * bandwidth KiB/s
    * throughput (ops/s)
    * latency (ms/op)
* Understand your workload
* Use EBS optimized instances

Certain factors can degrade HDD performance

* Initial access of blocks restored from snapshots
* Perf while a snapshot is being taken

Initializing

* New ECS volume? Ready to go, no perwarming required
* New volume from snapshot - read all blocks with data before using in production. THis is called initializing.

Kernel

* Use a modern Linux kernel that supports indirect descriptors (3.11 and later)
* Increase read-ahead for high throughput, read heavy workloads on throughput optimized hdd and cold hdd volumes.
    * Set as per volume configuration only for HDD volumes

RAID

* Can combine volumes into a RAID
* Use RAID 0 when:
    * Storage requirement > 16 TiB
    * Throughput requirement > 500 MB/s
    * IOPS requirement > 32,000 at 16K
* Avoid RAID redundancy because:
    * EBS data is already distributed
    * RAID 1 cuts available EBS bandwidth in half
    * RAID 5 and 6 loses 20% - 30% of usable I/O to parity.

CloudWatch Metrics - Use to [Monitor Volume Status](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring-volume-status.html)

* VolumeReadBytes
* VolumeWriteBytes
* VolumeReadOps
* VolumeWriteOps
* VoldumeQueueLength

Can use the `get-metric-data` and `get-metric-statistics` commands.