import torch
import matplotlib.pyplot as plt

def plot_histories(models):
    # Stack histories into tensors
    train_histories = torch.tensor([model.tlosshistory for model in models])
    val_histories = torch.tensor([model.vlosshistory for model in models])
    
    
    # Ensure everything is float64 for plotting precision
    train_mean = train_histories.mean(0).numpy()
    #train_std = train_histories.std(0).numpy()
    
    val_mean = val_histories.mean(0).numpy()
    #val_std = val_histories.std(0).numpy()
    
    steps = models[0].steps

    _, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # Plot mean lines
    ax.plot(steps, train_mean, color='tab:blue', linewidth=3, label='Train')
    ax.plot(steps, val_mean, color='tab:green', linewidth=3, label='Validation')

    # Apply to plot

    ax.set_xlabel("Epoch", fontsize="20")
    ax.set_ylabel("Loss", fontsize="20")
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='best', fontsize="20")
    plt.tight_layout()
    plt.show()


def plot_predictions(pred, true, t, dpi=100):
    """Plots predictions versus true solutions"""

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=dpi, sharex=True)

    labels = ['Prediction', 'Ground Truth']
    handles = []

    for k in range(2):
        for l in range(pred.shape[0]):
            # True values
            line_true, = axes[k].plot(t, true[l, :, k].detach().numpy(), color='tab:blue', alpha=0.6, linewidth=4)
            # Predictions
            line_pred, = axes[k].plot(t, pred[l, :, k].detach().numpy(), linestyle='--', color='tab:red', linewidth=2)

            if l == 0:
                handles = [line_pred, line_true]  # Save for legend (just once)

        axes[k].set_ylabel(r'$p(t)$' if k == 0 else r'$q(t)$', fontsize=12)
        axes[k].grid(True)

    axes[1].set_xlabel("Time", fontsize=20)

    # Only show x-ticks on the bottom
    axes[0].tick_params(labelbottom=False)

    # Add legend inside the upper subplot
    axes[0].legend(handles, labels, loc='upper right', fontsize=10, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()
    
    
    
