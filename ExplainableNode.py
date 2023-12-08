import torch

class ExplainableNode:
    def __init__(self):
        self.activation_values = {}
        self.gradient_sums = {}
        self.hooks = []
        self.batch_metrics = []  # Store metrics for each batch

    def register_hooks(self, model, layers):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for layer in layers:
            forward_hook = layer.register_forward_hook(self.forward_hook_fn)
            backward_hook = layer.register_backward_hook(self.backward_hook_fn)
            self.hooks.extend([forward_hook, backward_hook])

    def forward_hook_fn(self, module, input, output):
        self.activation_values[module] = output.detach()

    def backward_hook_fn(self, module, grad_input, grad_output):
        grad = grad_output[0].detach()
        self.gradient_sums[module] = grad.sum(dim=0)

    def calculate_metrics(self):
        # This function now returns only the metric values for simplicity
        metrics = {}
        for layer in self.activation_values.keys():
            act = self.activation_values[layer]
            grad_sum = self.gradient_sums[layer]
            gis = torch.mul(act, grad_sum).mean().item()
            metrics[str(layer)] = gis  # Convert layer to string to identify it
        self.batch_metrics.append(gis)
        return metrics

    def clear_values(self):
        self.activation_values.clear()
        self.gradient_sums.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

