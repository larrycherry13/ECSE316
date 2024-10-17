# ECSE 316 Assignment 1

**Authors**:  
- Laurent Chiricota (ID: 261113415)  
- Samy Sabir (ID: 261119166)  

**Python Version**: 3.11.5

## Project Description
This assignment implements a DNS client in Python. The client allows users to query DNS servers for information such as IP addresses, mail servers, and name servers.

## Usage

To run the program, use the following command structure:

python DnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name

- `timeout`: (Optional) The time to wait for a response, in seconds. Default is 5 seconds.
- `max-retries`: (Optional) The maximum number of retry attempts if the DNS query fails. Default is 3 retries.
- `port`: (Optional) The port number for the DNS server. Default is 53.
- `-mx`: (Optional) Request the mail server (MX record) of the domain.
- `-ns`: (Optional) Request the name server (NS record) of the domain.
- `@server`: The DNS server to query (e.g., 8.8.8.8 for Google's DNS server).
- `name`: The domain name to resolve (e.g., example.com).