# Modifying EBS Volumes


You can perform certain dynamic operations on exisint volumes with no downtime or negative performance effects:

* *Increase* volume capacity
* Change volume type
* Increase or decrease IOS

You can combine cloudwatch and lamdba to trigger the above modifications.

Modify steps:

* Modify
    * States - modifying, optimizing, complete. 
* Monitor
* Extend
    * If the size of the volume was modified, extend the volume's file system to take advantage of the additional capacity
    * You can resize the volume when it reaches the optimizing state.
* Use the `modify-volume` and `describe-volume-modifications` commands

Extend the file system using an OS specific command, for example:

* Use resize2fs for ext2, ext3, and ext4 files systems
* Use xfs_growfs for xfs file systems