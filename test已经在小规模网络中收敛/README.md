# Repository: partial_gradient_federated_split
# Files included in this one document (use as separate files if you prefer):
# - model.py
# - client.py
# - server.py
# - run.py
# - README.md
# -------- README.md --------
# Partial-gradient federated simulation

This repository implements the defensive scheme you described: clients only upload gradients for the last-n (closest-to-output) layers; the server averages these last-n gradients and updates its local last-n model; clients then receive updated last-n parameters and locally update their earlier layers while keeping last-n frozen.

Requirements:
- Python 3.8+
- torch, torchvision

Install:
```
pip install torch torchvision
```

Run:
```
python run.py --num_clients 3 --rounds 10 --cut_n 2
```

Notes:
- This is a simplified simulation for research/experimentation. It is NOT optimized for performance or production use.
- Names of parameters must align between client and server models. The simple design in `SimpleCNN_Full` ensures that.
