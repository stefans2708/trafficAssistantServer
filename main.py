import base64
import socket
import threading

import cv2
import numpy as np

PORT = 9999
HOST = '192.168.0.21'
ADDRESS = (HOST, PORT)
HEADER_LENGTH = 64
FORMAT = 'utf-8'

"""
If client disconnects, sometimes may happen that server still count that client as connected.
So, if client wants to reconnect, server will deny that because he still thinks that that client is connected.
Thus, we use this message to sent from client to server when client is disconnecting.

Should check also check:  socket.setsocketopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1). 
This probably allows reusing address and reconnecting.
"""
DISCONNECT_MSG = '!DISCONNECT'
BUFF_SIZE = 65536

# SOCK_STREAM is for TCP
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

# bind server to this address; every request that hits this
# combination of address and port will access to the socket
server.bind(ADDRESS)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        print("Waiting...")
        packet, _ = conn.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet, ' /')
        npdata = np.fromstring(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, 1)
        cv2.imshow("Received image", frame)

    conn.close()


# def handle_client(conn, addr):
#     print(f"[NEW CONNECTION] {addr} connected.")
#
#     connected = True
#     while connected:
#         print("Waiting...")
#         msg = conn.recv(4).decode(FORMAT)
#         if msg:
#             print(f"[{addr}]: {msg}")
#
#     conn.close()


# def handle_client(conn, addr):
#     print(f"[NEW CONNECTION] {addr} connected.")
#
#     connected = True
#     while connected:
#         msg_length = conn.recv(HEADER_LENGTH).decode(FORMAT)
#         if msg_length:
#             msg_length = int(msg_length)
#             msg = conn.recv(msg_length).decode(FORMAT)
#             if msg == DISCONNECT_MSG:
#                 connected = False
#             print(f"[{addr}]: {msg}")
#
#     conn.close()


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {HOST}")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")  # One thread is running always so we subtract 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            server.close()
            break


print('[STARTING] Server is starting...')
start()
