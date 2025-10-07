import numpy as np
from util.alif_eligibility_propagation import CustomALIF, exp_convolve

def get_cell(tag, n_input, FLAGS):

    taua = FLAGS["tau_a"]
    tauv = FLAGS["tau_v"]

    n_out = FLAGS["n_regular"] + FLAGS["n_adaptive"]
    thr_new = FLAGS["thr"] / (1 - np.exp(-FLAGS["dt"] / tauv)) if np.isscalar(tauv) else \
        [FLAGS["thr"] / (1 - np.exp(-FLAGS["dt"] / tv)) for tv in tauv]
    
    beta_new = FLAGS["beta"] * (1 - np.exp(-FLAGS["dt"] / taua)) / (1 - np.exp(-FLAGS["dt"] / tauv))
    print("565 new threshold = {:.4g}\n565 new beta      = {:.4g}".format(thr_new, beta_new))
    beta_new = np.concatenate([np.zeros(FLAGS["n_regular"]), np.ones(FLAGS["n_adaptive"]) * beta_new])
    

    return CustomALIF(n_in=n_input, n_rec=n_out, tau=tauv,
                      dt=FLAGS["dt"], tau_adaptation=taua, beta=beta_new, thr=thr_new,
                      dampening_factor=FLAGS["dampening_factor"],
                      tag=tag, n_refractory=FLAGS["n_ref"],
                      stop_gradients=FLAGS["eprop"] is not None, rec=FLAGS["rec"]
                      )