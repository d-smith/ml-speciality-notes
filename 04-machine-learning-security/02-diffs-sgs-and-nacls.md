# Differences Between Security Groups and NACLs

Subnets can have optional NACLs that can filter on IP range, port, and protocol.

NACLs - have inbound rulesets and outbound rulesets, stateless.

Security groups - stateful. Security groups remember and allow return traffic. For example outbound traffic allowed from an EC2 instance will have the response allowed back in.

* Packet to exit, SG examined.
* Packet hits outbound subnet - allowed out?
* Packet wants to enter next subnet - allowed in?
* Wants to enter an instance - security group check?
* Response - no check
* Subnet exit - check (passport control)
* Subnet entrance - check (passport control)
* Response into originating instance - no check

Why not use NACLs for everything?

* Only checked at subnet boundaries
