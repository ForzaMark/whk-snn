import numpy as np
import torch
from util.utils import train, compute_classification_accuracy, get_train_test_data, get_weights, get_device, save_history_plot

# The coarse network structure and the time steps are dicated by the SHD dataset. 
nb_inputs  = 700
nb_hidden  = 200
nb_outputs = 20

time_step = 1e-3
nb_steps = 100
max_time = 1.4

batch_size = 256
dtype = torch.float

tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

device = get_device()
nb_epochs = 50

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_train_test_data()    

    weights = get_weights(device, 
                          nb_inputs=nb_inputs,
                          nb_hidden=nb_hidden,
                          nb_outputs=nb_outputs,
                          dtype=dtype)

    loss_hist = train(x_train, 
                      y_train, 
                      weights=weights,
                      batch_size=batch_size,
                      nb_steps=nb_steps,
                      nb_inputs=nb_inputs,
                      nb_hidden=nb_hidden,
                      nb_outputs=nb_outputs,
                      device=device,
                      dtype=dtype,
                      alpha=alpha,
                      beta=beta,
                      nb_epochs=nb_epochs,
                      max_time=max_time,
                      lr=2e-4)
    
    save_history_plot(loss_hist, name='zenker_loss')
    print("Training done.")
    print("Training accuracy: %.3f"%(compute_classification_accuracy(x_train,y_train, batch_size=batch_size, nb_steps=nb_steps, nb_inputs=nb_inputs, max_time=max_time, device=device, nb_hidden=nb_hidden, nb_outputs=nb_outputs, dtype=dtype, alpha=alpha, beta=beta, weights=weights)))
    print("Test accuracy: %.3f"%(compute_classification_accuracy(x_test,y_test, batch_size=batch_size, nb_steps=nb_steps, nb_inputs=nb_inputs, max_time=max_time, device=device, nb_hidden=nb_hidden, nb_outputs=nb_outputs, dtype=dtype, alpha=alpha, beta=beta, weights=weights)))   