import matplotlib.pyplot as plt


def save_history_plot(history, path):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Array Plot")
    plt.grid(True)

    plt.savefig(path)
    plt.close()

def save_loss_per_time_step_plot(epoch_loss_per_time_step, path):
    x = range(100)
    ys_last = [(value["epoch"], value["loss_per_time_step_last_element"]) for value in epoch_loss_per_time_step]
    ys_first = [(value["epoch"], value["loss_per_time_step_first_element"]) for value in epoch_loss_per_time_step]
    ys_averaged = [(value["epoch"], value["loss_per_time_step_averaged_element"]) for value in epoch_loss_per_time_step]

    ys = [
        ys_last,
        ys_first,
        ys_averaged
    ]

    names = [
        'Last Elements Loss of Training Epoch',
        'First Elements Loss of Training Epoch',
        'Averaged Elements Loss of Training Epoch'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for i, ax in enumerate(axes):
        for epoch, loss_per_time_step in ys[i]:
            ax.set_title(names[i])
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Loss')
            if i == 0:
                ax.plot(x, loss_per_time_step, label=f'Epoch {epoch}')
            else:
                ax.plot(x, loss_per_time_step)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2))

    plt.savefig(path)
    plt.close()