import matplotlib.pyplot as plt
import torch

def plot_and_save_losses(train_losses, dev_losses, save_path="loss_curves.png"):
    """Plot and save loss_curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(dev_losses, label="dev Loss", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and dev Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Loss plot saved to {save_path}")

def save_best_model(model, dev_loss, best_dev_loss, epoch):
    """Save model if it has the best dev loss so far"""
    if dev_loss < best_dev_loss:
        
        torch.save(model.state_dict(), f"best_model.pt")
        
        print(f"New best model saved at epoch {epoch}! Dev loss: {dev_loss:.4f}")
        return dev_loss
    return best_dev_loss

def check_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")