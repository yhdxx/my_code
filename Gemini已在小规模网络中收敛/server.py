# server.py

import socket
import torch
import torch.optim as optim
import pickle
from utils import get_server_model
import struct

HOST = '127.0.0.1'
PORT = 65432

# Model parameters
N = 3  # Number of layers in the server model
server_model = get_server_model(N)
optimizer = optim.SGD(server_model.parameters(), lr=0.01)


# Helper function to receive data with a header
def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the data
    return recvall(sock, msglen)


# Helper function to receive all data
def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def handle_client_connection(conn):
    print("Client connected.")

    for i in range(10):  # 10 global iterations
        print(f"Server - Global Iteration {i + 1}")
        try:
            # 1. Receive gradients from the client
            full_data = recv_msg(conn)
            if full_data is None:
                print("Client disconnected.")
                break

            gradients_bytes = pickle.loads(full_data)

            # Apply gradients to the server model
            optimizer.zero_grad()
            for (param, grad) in zip(server_model.parameters(), gradients_bytes):
                if grad is not None:
                    param.grad = grad

            # 2. Update the server model
            optimizer.step()

            # 3. Send the updated parameters back to the client
            updated_params = [param.detach().clone() for param in server_model.parameters()]
            msg = pickle.dumps(updated_params)
            # Prepend message with its length
            msg = struct.pack('>I', len(msg)) + msg
            conn.sendall(msg)

            print("Server received gradients and sent back updated parameters.")
        except (socket.error, pickle.UnpicklingError) as e:
            print(f"Error handling connection: {e}")
            break


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            handle_client_connection(conn)


if __name__ == '__main__':
    start_server()