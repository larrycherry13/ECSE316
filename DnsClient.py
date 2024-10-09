import socket
import struct
import time
import argparse
import random
import sys

# The following function is used to parse the arguments from the command line
def parse_arguments():
    parser = argparse.ArgumentParser(description="DNS Client Arguments Parser")

    #Optional arguments
    parser.add_argument('-t', type=int, default=5, help="Timeout in seconds. Default: 5")
    parser.add_argument('-r', type=int, default=3, help="Max retries. Default: 3")
    parser.add_argument('-p', type=int, default=53, help="UDP port number. Default: 53)")

    #Flags for Query Types
    parser.add_argument('-mx', action='store_true', help="Query MX record")
    parser.add_argument('-ns', action='store_true', help="Query NS record")

    #SERVER AND DOMAIN NAME
    parser.add_argument('server', type=str, help="DNS server IP address")
    parser.add_argument('domain_name', type=str, help="Domain name to query")
    
    args = parser.parse_args()
    query_type = 'A'
    if args.mx:
        query_type = 'MX'
    elif args.ns:
        query_type = 'NS'
    
    return {
        'timeout': args.t,
        'max_retries': args.r,
        'port': args.p,
        'query_type': query_type,
        'server': args.server.strip('@'),
        'domain_name': args.domain_name
    }

# Build the DNS query from the domain name
def build_dns_query(domain_name, query_type):
    # Generate a random 16-bit transaction ID
    transaction_id = random.randint(0, 0xFFFF)  # Correctly generate a 16-bit transaction ID
    fields = 0x0100  # Standard recursive query
    qdcount = 1     # Number of questions
    ancount = 0     # Number of answers (in the query)
    nscount = 0     # Number of authority resource records (not used here)
    arcount = 0     # Number of additional resource records (not used here)
    
    # DNS header
    header = struct.pack('!HHHHHH', transaction_id, fields, qdcount, ancount, nscount, arcount)
    
    # Convert the domain name to DNS format
    qname = b''
    for part in domain_name.split('.'):
        qname += struct.pack('!B', len(part)) + part.encode('utf-8')
    qname += b'\x00'  # Null byte to terminate the domain name
    
    # Query Type (qtype: A = 1, NS = 2, MX = 15) and Class (1 = IN for Internet)
    question = struct.pack('!HH', query_type, 1)
    
    return header + qname + question

# Function to send the DNS query and receive a response
def send_dns_query(server, port, query, timeout, retries):
    #UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #set timeout
    sock.settimeout(timeout)
    
    for attempt in range(retries):
        try:
            start_time = time.time()
            #Send to query server
            sock.sendto(query, (server, port))

            response, _ = sock.recvfrom(512)  # Standard size for DNS response
            end_time = time.time()
            
            time_taken = end_time - start_time
            
            return response, time_taken, attempt
        
        except socket.timeout:
            print(f"Attempt {attempt+1} failed, retrying...")
    
    return None, None, retries

# Parse the DNS response
def parse_dns_response(response, query_type):
    transaction_id = struct.unpack('!H', response[:2])[0]
    flags = struct.unpack('!H', response[2:4])[0]
    qdcount = struct.unpack('!H', response[4:6])[0]
    ancount = struct.unpack('!H', response[6:8])[0]
    arcount = struct.unpack('!H', response[10:12])[0]

    # Skip question section
    response = response[12:]

    # Parsing Answer section
    if ancount > 0:
        print(f"***Answer Section ({ancount} records)***")
        for _ in range(ancount):
            # Reading the answer (skipping the domain name for simplicity)
            response = response[2:]  # Skipping pointer or name
            rtype, rclass, ttl, rdlength = struct.unpack('!HHIH', response[:10])
            rdata = response[10:10+rdlength]
            response = response[10+rdlength:]

            # Handling A, MX, NS, and CNAME types
            if rtype == 1:  # A record (IP address)
                ip_address = '.'.join(map(str, rdata))
                print(f"IP\t{ip_address}\t{ttl}\t{'auth' if flags & 0x0400 else 'nonauth'}")
            elif rtype == 15:  # MX record
                preference = struct.unpack('!H', rdata[:2])[0]
                exchange = rdata[2:].decode('utf-8')
                print(f"MX\t{exchange}\t{preference}\t{ttl}\t{'auth' if flags & 0x0400 else 'nonauth'}")
            elif rtype == 2:  # NS record
                ns = rdata.decode('utf-8')
                print(f"NS\t{ns}\t{ttl}\t{'auth' if flags & 0x0400 else 'nonauth'}")
            elif rtype == 5:  # CNAME record
                cname = rdata.decode('utf-8')
                print(f"CNAME\t{cname}\t{ttl}\t{'auth' if flags & 0x0400 else 'nonauth'}")
    else:
        print("NOTFOUND")

    # Parsing Additional section if exists
    if arcount > 0:
        print(f"***Additional Section ({arcount} records)***")
        # Implement parsing for Additional Section if needed

# Print query info
def print_query_info(domain_name, server, query_type):
    print(f"DnsClient sending request for {domain_name}")
    print(f"Server: {server}")
    request_type = "A" if query_type == 1 else ("MX" if query_type == 15 else "NS")
    print(f"Request type: {request_type}")

# Main execution




if __name__ == "__main__":

    send_dns_query()




  
    
    
    
  
