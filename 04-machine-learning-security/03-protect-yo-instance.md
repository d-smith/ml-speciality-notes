# Protect Your Instance With Security Groups

Every instance in your VPC has a security group around it.

* Think of it as a firewall around each individual instance that by default blocks all incoming traffic.
* Author security groups to authorize traffic in - port, protocol, IP address range
* Stack security groups to control traffic
    * e.g. internet to web servers only, web servers to app servers only, app server to db only, no way to go outside to app server, or outside to db.
    * Build layers of protection
    

