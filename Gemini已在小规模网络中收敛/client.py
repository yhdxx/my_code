# client.py

import socket
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from utils import get_model, get_data_loaders
import struct

HOST = '127.0.0.1'
PORT = 65432

# Model and training parameters
MI = 5
CUT_N = 2
N = MI - CUT_N  # Server model layers
EPOCHS = 2  # Local epochs
LOCAL_LR = 0.01


# Helper function to send data with a header
def send_msg(sock, msg):
    # Prepend message with its length
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


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


def start_client():
    train_loaders, _ = get_data_loaders(num_clients=1, batch_size=32)
    client_loader = train_loaders[0]

    # Client model has mi layers
    client_model = get_model(MI, CUT_N)
    criterion = nn.CrossEntropyLoss()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to server.")

        for i in range(10):  # 10 global iterations
            print(f"Client - Global Iteration {i + 1}")

            # Phase 1: Train the last n layers
            # This is the "server-side" part of the client model
            client_shared_params = list(client_model.parameters())[-(N * 2):]  # n layers, 2 params/layer
            optimizer_shared = optim.SGD(client_shared_params, lr=LOCAL_LR)

            for epoch in range(EPOCHS):
                for inputs, labels in client_loader:
                    optimizer_shared.zero_grad()

                    # Forward pass: client-only part
                    client_only_output = client_model.fc_client_only(client_model.flatten(
                        client_model.pool2(client_model.relu2(client_model.conv2(
                            client_model.pool1(client_model.relu1(client_model.conv1(inputs))))))))

                    # Forward pass: shared part
                    outputs = client_model.fc3(
                        client_model.relu4(client_model.fc2(client_model.relu3(client_model.fc1(client_only_output)))))

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_shared.step()

            # 2. Extract and upload gradients of the last n layers
            shared_gradients = [param.grad.detach().clone() if param.grad is not None else None for param in
                                client_shared_params]
            msg = pickle.dumps(shared_gradients)
            send_msg(s, msg)

            # 3. Receive the updated parameters from the server
            full_data = recv_msg(s)
            if full_data is None:
                print("Server disconnected.")
                break
            updated_params = pickle.loads(full_data)

            # Update the client's shared part with server-sent parameters
            for local_param, server_param in zip(client_shared_params, updated_params):
                local_param.data.copy_(server_param.data)

            print("Client received updated parameters from server and updated model.")

            # Phase 2: Train the first mi-n layers
            client_only_params = list(client_model.parameters())[:-(N * 2)]
            optimizer_client_only = optim.SGD(client_only_params, lr=LOCAL_LR)

            for epoch in range(EPOCHS):
                for inputs, labels in client_loader:
                    optimizer_client_only.zero_grad()

                    # Forward pass
                    outputs = client_model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_client_only.step()

            # To show some progress
            with torch.no_grad():
                # We need to get a new batch here
                test_inputs, test_labels = next(iter(client_loader))
                outputs = client_model(test_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total = test_labels.size(0)
                correct = (predicted == test_labels).sum().item()
                print(f"Client - Local Accuracy: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    start_client()