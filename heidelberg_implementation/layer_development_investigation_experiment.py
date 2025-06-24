from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_simplified_snn import train_simplified_snn
from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, BETA, TIME_STEPS

best_sparsity = 0
best_number_hidden_layer = 2
best_number_hidden_neurons = 3000

num_epochs = 14

def create_best_grid_search_model(sparsity = 0.0):
    return ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=best_number_hidden_neurons,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=best_number_hidden_layer,
                                                 sparsity=sparsity)

best_grid_search_model = create_best_grid_search_model()
best_grid_search_model_80_percent_of_connections = create_best_grid_search_model(sparsity=0.2)
best_grid_search_model_50_percent_of_connections = create_best_grid_search_model(sparsity=0.5) 
best_grid_search_model_20_percent_of_connections = create_best_grid_search_model(sparsity=0.8)
best_grid_search_model_5_percent_of_connections = create_best_grid_search_model(sparsity=0.95)

if __name__ == '__main__':
    train_simplified_snn(best_grid_search_model, 
                        num_epochs=num_epochs, 
                        save_model='./models/experiment_layer_development_investigation/best_grid_search', 
                        save_plots='./output/experiments_layer_development_investigation/best_grid_search', 
                        additional_output_information={
                            'num_hidden_layer': best_number_hidden_layer,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0
                        },
                        output_file_path='./output/experiments_layer_development_investigation/best_grid_search.json')
    
    train_simplified_snn(best_grid_search_model_80_percent_of_connections, 
                        num_epochs=num_epochs, 
                        save_model='./models/experiment_layer_development_investigation/best_grid_search_80_percent_connections', 
                        save_plots='./output/experiments_layer_development_investigation/best_grid_search_80_percent_connections', 
                        additional_output_information={
                            'num_hidden_layer': best_number_hidden_layer,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0.2
                        },
                        output_file_path='./output/experiments_layer_development_investigation/best_grid_search_80_percent_connections.json')
    
    train_simplified_snn(best_grid_search_model_50_percent_of_connections, 
                        num_epochs=num_epochs, 
                        save_model='./models/experiment_layer_development_investigation/best_grid_search_50_percent_connections', 
                        save_plots='./output/experiments_layer_development_investigation/best_grid_search_50_percent_connections', 
                        additional_output_information={
                            'num_hidden_layer': best_number_hidden_layer,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0.5
                        },
                        output_file_path='./output/experiments_layer_development_investigation/best_grid_search_50_percent_connections.json')
    
    train_simplified_snn(best_grid_search_model_20_percent_of_connections, 
                        num_epochs=num_epochs, 
                        save_model='./models/experiment_layer_development_investigation/best_grid_search_20_percent_connections', 
                        save_plots='./output/experiments_layer_development_investigation/best_grid_search_20_percent_connections', 
                        additional_output_information={
                            'num_hidden_layer': best_number_hidden_layer,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0.8
                        },
                        output_file_path='./output/experiments_layer_development_investigation/best_grid_search_20_percent_connections.json')
    
    train_simplified_snn(best_grid_search_model_5_percent_of_connections, 
                        num_epochs=num_epochs, 
                        save_model='./models/experiment_layer_development_investigation/best_grid_search_5_percent_connections', 
                        save_plots='./output/experiments_layer_development_investigation/best_grid_search_5_percent_connections', 
                        additional_output_information={
                            'num_hidden_layer': best_number_hidden_layer,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0.95
                        },
                        output_file_path='./output/experiments_layer_development_investigation/best_grid_search_5_percent_connections.json')