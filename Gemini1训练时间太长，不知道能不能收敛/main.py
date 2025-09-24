import torch
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from client import Client
from server import Server
from model import ClientNet, ServerNet

# Set up logging and data collection
logging_data = []


# Function to run the simulation for a given server layer count
def run_simulation(server_layers, num_rounds=1000):
    # client_mi 是总层数，server_n 是服务器和客户端共享的层数
    client_mi = 100
    server_n = 3  # 固定服务器模型为3层

    global_client_model = ClientNet(mi=client_mi)
    server = Server(n=server_n, num_clients=10)

    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Split the dataset for 10 clients (for simplicity, a random split)
    client_datasets = [torch.utils.data.Subset(trainset, range(i * 6000, (i + 1) * 6000)) for i in range(10)]

    clients = [Client(i, global_client_model, client_datasets[i]) for i in range(10)]

    # Create a test dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    losses = []

    for round_num in range(num_rounds):
        # Select clients randomly
        selected_clients = random.sample(clients, 5)

        # Train clients and collect gradients
        client_grads_list = []
        client_losses = []
        for client in selected_clients:
            grads, loss = client.train_and_get_partial_gradient(server.server_model)
            client_grads_list.append(grads)
            client_losses.append(loss)

        avg_loss = sum(client_losses) / len(client_losses)
        losses.append(avg_loss)

        # Server aggregates gradients
        aggregated_grads = server.aggregate_gradients(client_grads_list)

        # Server updates its model and gets updated parameters
        updated_server_model = server.update_and_get_params(aggregated_grads)

        # Distribute updated server model parameters back to clients
        for client in clients:
            client.model.back_layers.load_state_dict(updated_server_model.layers.state_dict())

        # Evaluate the global client model periodically (e.g., every 50 rounds)
        if (round_num + 1) % 50 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = global_client_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(
                f'Server Layers: {server_layers}, Round: {round_num + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Final evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = global_client_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    final_accuracy = 100 * correct / total

    logging_data.append({'server_layers': server_layers, 'losses': losses, 'final_accuracy': final_accuracy})


# Run simulations for different server layer counts
server_layer_counts = list(range(1, 101))
for n_layers in server_layer_counts:
    print(f"\n--- Running simulation for server with {n_layers} layers ---")
    run_simulation(n_layers)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plotting final accuracy
accuracies = [data['final_accuracy'] for data in logging_data]
ax1.set_xlabel('Server Layers (n)')
ax1.set_ylabel('Final Accuracy (%)', color='tab:blue')
ax1.plot(server_layer_counts, accuracies, 'o-', color='tab:blue', label='Final Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plotting a single example of loss curve for n=10
ax2 = ax1.twinx()
# Find the data for a specific n_layers, e.g., n=10
loss_curve_data = next((data for data in logging_data if data['server_layers'] == 10), None)
if loss_curve_data:
    ax2.set_ylabel('Training Loss', color='tab:red')
    ax2.plot(range(1, 1001), loss_curve_data['losses'], color='tab:red', label='Training Loss (n=10)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Performance vs. Server Layers')
plt.grid(True)
fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
plt.show()