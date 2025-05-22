from train_simplified_snn import num_hidden, num_inputs, num_outputs, beta, time_steps, get_device, train_simplified_snn
from varying_hidden_layer_1000_neurons_net import VaryingHiddenLayer1000NeuronsNet
from single_hidden_layer_1000_neurons_net import SingleHiddenLayer1000NeuronsNet

NUM_HIDDEN_LAYERS = [
    1,
    2,
    3,
    4
]

device = get_device()
num_epochs = 30
sparsity = 0

for number_hidden_layer in NUM_HIDDEN_LAYERS:
    print(f'Number hidden layer = {number_hidden_layer}')
    model = VaryingHiddenLayer1000NeuronsNet(num_input=num_inputs, 
                                             num_hidden=num_hidden, 
                                             num_output=num_outputs, 
                                             beta=beta, 
                                             time_steps=time_steps, 
                                             num_hidden_layers=number_hidden_layer)
    
    train_simplified_snn(model, 
                         num_epochs=num_epochs, 
                         additional_output_information={
                          'number_hidden_layer': number_hidden_layer   
                         },
                         output_file_path=f'./output/experiments_multiple_hidden_layer/number_hidden_layer_{number_hidden_layer}.json')


model = SingleHiddenLayer1000NeuronsNet(num_inputs=num_inputs, 
                                        num_outputs=num_outputs, 
                                        beta=beta, 
                                        time_steps=time_steps, 
                                        sparsity=sparsity)

train_simplified_snn(model, 
                     num_epochs=num_epochs,
                     output_file_path=f'./output/experiments_multiple_hidden_layer/baseline_single_hidden_layer.json')



