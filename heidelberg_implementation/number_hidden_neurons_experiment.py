from train_simplified_snn import num_inputs, num_outputs, beta, time_steps, get_device, train_simplified_snn
from neural_nets.varying_hidden_layer_1000_neurons_net import VaryingHiddenLayer1000NeuronsNet
import numpy as np

NUM_HIDDEN_NEURONS = np.arange(500, 5500, 500)

device = get_device()
num_epochs = 30
sparsity = 0

for number_hidden_neurons in NUM_HIDDEN_NEURONS:
    print(f'Number hidden neurons = {number_hidden_neurons}')
    model = VaryingHiddenLayer1000NeuronsNet(num_input=num_inputs, 
                                            num_hidden=number_hidden_neurons,
                                            num_output=num_outputs, 
                                            beta=beta, 
                                            time_steps=time_steps,
                                            num_hidden_layers=1)

    train_simplified_snn(model, 
                        num_epochs=num_epochs,
                        save_model=False,
                        save_plots=False,
                        additional_output_information={'number_hidden_neurons': int(number_hidden_neurons)},
                        output_file_path=f'./output/experiments_number_hidden_neurons/number_hidden_neurons_{number_hidden_neurons}.json')



