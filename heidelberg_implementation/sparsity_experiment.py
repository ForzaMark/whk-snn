import torch
from constants import (
    BETA,
    DEVICE,
    NUMBER_HIDDEN_LAYERS,
    NUMBER_HIDDEN_NEURONS,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn


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
    for sparsity in SPARSITIES:
        net = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                           number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                           number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                           number_output_neurons=NUMBER_OUTPUT_NEURONS, 
                                           beta=BETA, 
                                           threshold=THRESHOLD,
                                           time_steps=TIME_STEPS, 
                                           sparsity=sparsity).to(DEVICE)
        
        count_nonzero_weights(net)

        train_snn(net, num_epochs=30, 
                             additional_output_information={'sparsity': sparsity}, 
                             output_file_path=f'./output/experiments_sparsity/sparsity_{sparsity}.json')





