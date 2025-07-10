def compute_epoch_loss_per_time_step(loss_per_time_step_hist, epoch, time_steps):
    last_element = loss_per_time_step_hist[-1]
    first_element = loss_per_time_step_hist[0]
    averaged_element = []

    for i in range(time_steps):
        averaged_element.append(np.mean(np.array(loss_per_time_step_hist)[:, i]))

    return ({
        'epoch': epoch,
        'loss_per_time_step_last_element': last_element,
        'loss_per_time_step_first_element': first_element,
        'loss_per_time_step_averaged_element': averaged_element
    })
