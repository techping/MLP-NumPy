# Network Configuration
# Ziping Chen
# March 2020

net_conf = {
    'architecture': {
        'layer': {
            'input': {
                'num': 784,
            },
            'hidden': {
                'num': 48,
                'activation': 'relu',
            },
            'output': {
                'num': 10,
                'activation': 'softmax',
            },
        },
        'regularizer': ['l2', 0.001], #None,
    },
    'training': {
        'eta': 1.0,
        'optimizer': 'sgd',
        'epoch': 50,
        'minibatch': 500,
        'momentum': False,
        'decay': True, #False
    }
}