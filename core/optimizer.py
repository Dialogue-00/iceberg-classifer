from torch import optim


def get_opt(model, params):
    if params.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=params.learning_rate)
    if params.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=params.learning_rate)
    if params.optimizer == 'rms':
        opt = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99)
    if params.optimizer == 'moment':
        opt = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    # if params.optimizer == 'adamw':
    #     opt = optim.


    return opt