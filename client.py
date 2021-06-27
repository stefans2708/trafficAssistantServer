import socket

PORT = 9999
SERVER = '192.168.0.21'
ADDRESS = (SERVER, PORT)
HEADER_LENGTH = 64
FORMAT = 'utf-8'

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDRESS)


def send(msg):
    message = msg.encode(FORMAT)  # encodes message string to bytes
    msg_length = len(message)  # number of bytes in message
    send_length = str(msg_length).encode(FORMAT)  # message length in bytes
    send_length += b' ' * (HEADER_LENGTH - len(send_length))  # message length padded with bytes to have HEADER_LENGTH
    client.send(send_length)
    client.send(message)


send("Hello!")
