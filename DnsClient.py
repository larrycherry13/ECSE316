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
    query_type = 1 # A record default
    if args.mx:
        query_type = 15 #MX record
    elif args.ns:
        query_type = 2 #NS record
    
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
    
    return transaction_id, header + qname + question

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
def parse_dns_response(transaction_id, response, query_type):
    # Parse header
    response_id, flags, qdcount, ancount, nscount, arcount = struct.unpack('!HHHHHH', response[:12])

    # Check if the transaction ID matches
    if response_id != transaction_id:
        print("ERROR\tTransaction ID does not match. Exiting.")
        return

    # Check for flags
    auth = parse_flags(flags)

    if auth.startswith("ERROR"):
        print(auth)
        return
    elif auth == "NOTFOUND":
        print(auth)
        return

    # Check if the response has any answers
    if ancount == 0:
        print("NOTFOUND")
        return

    # Check if the response has the same number of questions as the query
    if qdcount != 1:
        print("ERROR\tInvalid number of questions [1] in the response. Exiting.")
        return

    offset = 12

    # Skip over the question section
    while response[offset] != 0:
        label_len = response[offset]
        offset += label_len + 1
    offset += 5  # Skip termination label (1 byte) + QTYPE (2 bytes) + QCLASS (2 bytes)

    # Parse the answer section
    print(f"*Answer Section ({ancount} records)*")
    
    offset = parse_records(response, offset, ancount, auth, "Answer")

    # Parse additional section if exists
    if arcount > 0:
        print(f"***Additional Section ({arcount} records)***")
        
        offset = parse_records(response, offset, arcount, auth, "Additional")

# Parse the flags in the DNS response
def parse_flags(flags):
    ERROR = ""

    # flag positions
    QR_MASK = 0x8000
    OPCODE_MASK = 0x7800
    AA_MASK = 0x0400  # Authoritative Answer
    TC_MASK = 0x0200
    RD_MASK = 0x0100
    RA_MASK = 0x8000
    Z_MASK = 0x0070
    RCODE_MASK = 0x000F

    # Extract flags
    is_response = (flags & QR_MASK) != 0
    opcode = (flags & OPCODE_MASK) >> 11
    is_authoritative = (flags & AA_MASK) != 0
    is_truncated = (flags & TC_MASK) != 0
    is_recursive_desired = (flags & RD_MASK) != 0
    is_recursive_available = (flags & RA_MASK) != 0
    z = (flags & Z_MASK) >> 4
    
    # Check if it’s a response
    if not is_response:
        ERROR += "ERROR\tNot a response packet"

    # Check if it’s a query
    if opcode != 0:
        ERROR += "ERROR\tNot a standard query"

    # Check if the response is truncated
    if is_truncated:
        ERROR += "ERROR\tResponse truncated"

    # Check if recursion is available
    if not is_recursive_available:
        ERROR += "ERROR\tRecursion not available"

    # Check if recursion was desired
    if not is_recursive_desired:
        ERROR += "ERROR\tRecursion not desired"

    # Chez z value
    if z != 0:
        ERROR += "ERROR\tZ value must be 0"

    #RCODE check
    rcode = flags & RCODE_MASK
    
    # Handling RCODE values
    response_codes = {
        0: "No error condition",
        1: "Format error: the name server was unable to interpret the query",
        2: "Server failure: the name server was unable to process this query due to a problem with the name server",
        3: "The domain name does not exist",
        4: "Not implemented: the name server does not support the requested kind of query",
        5: "Refused: the name server refuses to perform the specified operation for policy reasons"
    }
    
    # Check rcode
    if rcode != 0:
        if rcode == 3:
            if is_authoritative:
                ERROR = "NOTFOUND"
            else:
                ERROR += "NOTFOUND\tNon-authoritative response, domain may or may not exist."
        else:
            ERROR += f"ERROR\t{response_codes.get(rcode, 'ERROR' + chr(9) + f'Unknown error with RCODE {rcode}')}"
    
    if ERROR:
        return ERROR

    return "auth" if is_authoritative else "nonauth"

# Helper function to parse DNS records
def parse_records(response, offset, record_count, auth, section_type):
    for _ in range(record_count):
        if offset + 12 > len(response):
            print(f"ERROR\tIncomplete {section_type} record. Exiting.")
            return offset  # Return the offset to indicate failure

        # Skip name/pointer
        while True:
            if response[offset] == 0:
                offset += 1
                break
            if response[offset] & 0xC0 == 0xC0:
                offset += 2  # Skip pointer (2 bytes)
                break
            offset += 1

        res_type = int.from_bytes(response[offset:offset + 2], byteorder='big')
        offset += 2

        class_field = int.from_bytes(response[offset:offset + 2], byteorder='big')
        if class_field != 0x0001:
            print(f"ERROR\tIncorrect class field in the {section_type} section. Exiting.")
            return offset  # Return the offset to indicate failure
        offset += 2

        ttl = int.from_bytes(response[offset:offset + 4], byteorder='big')
        offset += 4

        rd_length = int.from_bytes(response[offset:offset + 2], byteorder='big')
        offset += 2

        if offset + rd_length > len(response):
            print(f"ERROR\tIncomplete {section_type} record data. Exiting.")
            return offset  # Return the offset to indicate failure

        # Handle the resource record type
        if res_type == 1:  # A record (IP address)
            rdata = response[offset:offset + rd_length]
            print(f"IP\t{'.'.join(map(str, rdata))}\t{ttl}\t{auth}")
        elif res_type == 2:  # NS record
            alias = parse_answer_data(response, offset)
            print(f"NS\t{alias}\t{ttl}\t{auth}")
        elif res_type == 5:  # CNAME record
            alias = parse_answer_data(response, offset)
            print(f"CNAME\t{alias}\t{ttl}\t{auth}")
        elif res_type == 15:  # MX record
            pref = int.from_bytes(response[offset:offset + 2], "big")
            alias = parse_answer_data(response, offset + 2)  # Skip preference field
            print(f"MX\t{alias}\t{pref}\t{ttl}\t{auth}")

        offset += rd_length  # Move offset forward by rd_length

    return offset

def parse_answer_data(response, offset):
    labels = []
    while True:
        length = response[offset]
        
        # Check if it’s a pointer
        if length & 0xC0 == 0xC0:
            # Pointer; offset is the next 14 bits
            pointer_offset = ((length & 0x3F) << 8) | response[offset + 1]
            labels.append(parse_answer_data(response, pointer_offset))
            offset += 2
            break
        elif length == 0:
            # End of the domain name
            offset += 1
            break
        else:
            # Read the label
            offset += 1
            labels.append(response[offset:offset + length].decode('utf-8'))
            offset += length
    return '.'.join(labels)


# Print query info
def print_query_info(domain_name, server, query_type):
    print(f"DnsClient sending request for {domain_name}")
    print(f"Server: {server}")
    request_type = "A" if query_type == 1 else ("MX" if query_type == 15 else "NS")
    print(f"Request type: {request_type}")

# Main execution




if __name__ == "__main__":

    args = parse_arguments()

    # Build the DNS query
    transaction_id, query = build_dns_query(args['domain_name'], args['query_type'])

    # Print query info
    print_query_info(args['domain_name'], args['server'], args['query_type'])

    # Send the DNS query and get a response
    response, time_taken, attempts = send_dns_query(
    args['server'], args['port'], query, args['timeout'], args['max_retries'])

    if response:
        # Parse the DNS response
        parse_dns_response(transaction_id, response, args['query_type'])
        print(f"Response received after {time_taken:.2f} seconds ({attempts} retries)")
    else:
        print("ERROR: Maximum retries exceeded.")








  
    
    
    
  
