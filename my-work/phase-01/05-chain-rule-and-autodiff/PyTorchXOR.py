import torch
import torch.nn as nn

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# MLP: 2 inputs -> 4 hidden neurons -> 1 output
model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# Training
for epoch in range(5000):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = model(X)

print("Predictions:")
print(predictions.numpy().round(3))

print("Classes:")
print((predictions >= 0.5).int().numpy())
