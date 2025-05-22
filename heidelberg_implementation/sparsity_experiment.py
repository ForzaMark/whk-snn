import torch
from single_hidden_layer_1000_neurons_net import SingleHiddenLayer1000NeuronsNet
from train_simplified_snn import train_simplified_snn, num_inputs, num_outputs, beta, time_steps, get_device

def count_nonzero_weights(model):
    nonzero = 0
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            tensor = param.data
            nonzero += torch.count_nonzero(tensor).item()
            total += tensor.numel()

    print(f"Non-zero weights: {nonzero} / {total} ({100 * nonzero / total:.2f}%)")


SPARSITIES = [
    0,
    0.2,
    0.6, 
    0.8
]

if __name__ == '__main__':
    device = get_device()

    for sparsity in SPARSITIES:
        net = SingleHiddenLayer1000NeuronsNet(num_inputs=num_inputs, num_outputs=num_outputs, beta=beta, time_steps=time_steps, sparsity=sparsity).to(device)
        
        count_nonzero_weights(net)

        train_simplified_snn(net, num_epochs=30, 
                             additional_output_information={'sparsity': sparsity}, 
                             output_file_path=f'./output/experiments_sparsity/sparsity_{sparsity}.json')





