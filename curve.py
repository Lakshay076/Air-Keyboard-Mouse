import matplotlib.pyplot as plt

# Data extracted directly from your logs
epochs = list(range(1, 26))

# "Loss" from your logs (Training Loss)
train_losses = [
    0.5173, 0.3995, 0.3898, 0.3853, 0.3803, 
    0.3757, 0.3724, 0.3686, 0.3647, 0.3606, 
    0.3573, 0.3535, 0.3488, 0.3445, 0.3400, 
    0.3352, 0.3293, 0.3241, 0.3180, 0.3110, 
    0.3042, 0.2968, 0.2901, 0.2837, 0.2798
]

# "Merged Val Acc" from your logs
val_accs = [
    88.07, 87.66, 89.00, 88.93, 88.14, 
    88.60, 89.19, 88.99, 88.82, 88.66, 
    89.17, 90.21, 89.94, 90.31, 90.52, 
    90.13, 89.89, 90.77, 90.98, 91.18, 
    91.36, 91.44, 91.56, 91.72, 91.69
]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Training Loss (Left Y-Axis)
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(epochs, train_losses, color=color, linewidth=2, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Create a second y-axis for Accuracy
ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Validation Accuracy (%)', color=color)  
ax2.plot(epochs, val_accs, color=color, linewidth=2, label='Val Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('ResNet-18 Training Dynamics: Loss vs Accuracy')
plt.tight_layout()

# Save it to the folder where your report expects it
import os
os.makedirs("screenshots", exist_ok=True)
plt.savefig('screenshots/resnet_loss.png')
print("✅ Graph saved as screenshots/resnet_loss.png")
plt.show()