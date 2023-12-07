import numpy as np
import matplotlib.pyplot as plt
import os

class ExplainableNode:
    def __init__(self, matrix_size=(4, 4), plot_interval=100):
        self.activation_values = []
        self.activation_derivatives = []
        self.matrix_size = matrix_size
        self.plot_interval = plot_interval
        self.figures = []
        self.plot_counter = 0
        self.page_number = 0

    def register_hooks(self, model):
        if hasattr(model, 'hooks'):
            for hook in model.hooks:
                hook.remove()
        model.hooks = []

        forward_hook = model.activation.register_forward_hook(
            lambda module, input, output: self.activation_values.append(output.detach().cpu().numpy())
        )
        backward_hook = model.activation.register_backward_hook(
            lambda module, grad_input, grad_output: self.activation_derivatives.append(grad_output[0].detach().cpu().numpy())
        )
        model.hooks.extend([forward_hook, backward_hook])

    def clear_values(self):
        self.activation_values.clear()
        self.activation_derivatives.clear()

    def plot_values(self, epoch, batch_idx, add_new_pages=False, keep_only_last_step=False):
        rows, cols = self.matrix_size
        max_plots_per_page = rows * cols

        if len(self.figures) == 0 or self.plot_counter >= max_plots_per_page:
            if add_new_pages or len(self.figures) == 0:
                fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
                self.figures.append((fig, axs.flatten()))
                self.plot_counter = 0
                self.page_number += 1
            else:
                return

        current_fig, current_axs = self.figures[-1]

        if keep_only_last_step:
            ax = current_axs[-1]
            ax.clear()
        else:
            ax = current_axs[self.plot_counter]
            self.plot_counter += 1

        if self.activation_values and self.activation_derivatives:
            ax.scatter(
                np.concatenate(self.activation_values).flatten(),
                np.concatenate(self.activation_derivatives).flatten(),
                color='red',
                marker='o'
            )
            ax.set_xlabel('Node Activation Values')
            ax.set_ylabel('Node Negative Derivative Values')
            ax.set_title(f'Epoch {epoch}, Batch {batch_idx}')
            ax.axvline(x=0, color='black', linewidth=2)
            ax.axhline(y=0, color='black', linewidth=2)
            ax.grid(True)

    def save_plots(self, epoch):
        for i, (fig, _) in enumerate(self.figures):
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.savefig(f'plots/plot_epoch{epoch}_page{i}.png')
            plt.close(fig)
        self.figures.clear()
