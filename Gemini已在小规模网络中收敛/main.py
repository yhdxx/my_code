import subprocess
import time
import os

if __name__ == '__main__':
    # Start the server in a new process
    print("Starting the server...")
    server_process = subprocess.Popen(['python', 'server.py'])

    # Wait a bit for the server to start
    time.sleep(5)

    # Start the client in a new process
    print("Starting the client...")
    client_process = subprocess.Popen(['python', 'client.py'])

    try:
        # Wait for processes to finish (or for manual interruption)
        client_process.wait()
        server_process.wait()
    except KeyboardInterrupt:
        print("Training interrupted. Terminating processes.")
        server_process.terminate()
        client_process.terminate()