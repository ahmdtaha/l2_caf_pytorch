def polynomial_lr_decay(global_step,
             init_learning_rate,
             max_iter,
             min_learning_rate=1e-5):
    power = 1
    lr = (init_learning_rate - min_learning_rate) * ((1 - global_step / max_iter) ** power) + min_learning_rate
    return lr