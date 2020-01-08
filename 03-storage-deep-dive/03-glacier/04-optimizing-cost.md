# Optimizing Costs on Amazon Glacier

Costs

* Much cheaper to store data on glacier than s3
* Retrieval pricing - based on expidited, standard, and bulk requests
* Request pricing - based on number of archive upload requests
* Data transfer pricing - based on data transferred in and out of Glacier
* Select pricing - based on amount of data scanned and number of select requests

Cost following - discrete cost reductions can be passed along to customers directly instead of hiding them behind an opaque all inclusive cost.

Targeting objects for transition to glacier

* Use storage class analysis to understand what data to move to glacier at what time
* Other factors to consider
    * Target objects that are 1MB or larger, note the 32KB overhead charge
    * Containerize small files into a tar or zip
    * How long the data will be stored  - glacier is for long lived data, minimum 90 days to retain, upload fees 10x of s3
     