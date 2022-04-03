from torch import optim


def get_opt(model, params):
    if params.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=params.learning_rate)
    if params.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5E-5)


    return opt