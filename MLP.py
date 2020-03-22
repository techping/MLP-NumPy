# MLP Class
# Ziping Chen
# March 2020
import numpy as np
import h5py
import matplotlib.pyplot as plt

from Layer import Layer

class MLP:
    
    def __init__(self, net_conf):
        self.arch = net_conf
        self.setup_network()
    
    def setup_network(self):
        assert len(self.arch['layer'].keys()) >= 3, "Network Architecture Error!!"
        self.net = {}
        for i, l in enumerate(self.arch['layer'].keys()):
            if i == 0:
                continue
            self.net[l] = Layer(self.arch['layer'][list(self.arch['layer'].keys())[i - 1]]['num'],
                                self.arch['layer'][l]['num'],
                                l,
                                self.arch['layer'][l]['activation'],
                                self.arch['regularizer'])
    
    # print network detail
    # fake keras interface
    def summary(self):
        print('Layer\tOutput\tActivation\tRegularization')
        for i, l in enumerate(self.arch['layer']):
            print('%s\t%5d\t%9s\t%14s' % (l, self.arch['layer'][l]['num'],
                'None' if not 'activation' in self.arch['layer'][l] else self.arch['layer'][l]['activation'],
                'None' if self.arch['regularizer'] is None or i == 0
                        else self.arch['regularizer'][0] + '=' + str(self.arch['regularizer'][1])))
    
    def feedforward(self, x):
        # feedforward
        for n in self.net:
            x = self.net[n].forward(x)
        return x
    
    def backpropagation(self, y_hat, y, eta):
        delta = y_hat - y
        # backpropagation
        for n in reversed(list(self.net.keys())):
            delta = self.net[n].backward(delta)
        # update parameters
        for n in self.net:
            self.net[n].update(eta)

    def cost(self, y_hat, y):
        # cross entropy
        # not use in this code
        return -np.sum(y * np.log(y_hat))
    
    # prediction
    # fake keras interface
    def predict(self, x):
        y_hat = self.feedforward(x)
        return y_hat
    
    def accuracy(self, x, y):
        y_hat = self.predict(x)
        pred = np.argmax(y_hat, axis=1)
        label = np.argmax(y, axis=1)
        return np.sum(pred == label) / pred.shape[0]
        
    # performance test
    # fake keras interface
    def perform(self, x, y):
        acc = self.accuracy(x, y)
        print("Accuracy: %lf" % (acc))

    # stochastic gradient descent
    def sgd(self, eta=0.001, epoch=50, minibatch=500, momentum=False, decay=False):
        # SGD
        # TODO:
        # momentum, adam, batch norm...
        iteration = self.training['xdata'].shape[0] // minibatch
        print("<====start training====>")
        print("batch size is", minibatch)
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        self.decay_pts = []
        for i in range(epoch):
            # 1. randomly shuffle
            p = np.random.permutation(self.training['xdata'].shape[0])
            for j in range(iteration):
                string = "\r[epoch %d/%d] iteration#%d " % (i + 1, epoch, j + 1)
                y_hat = self.feedforward(self.training['xdata'][p][j * minibatch:(j + 1) * minibatch])
                self.backpropagation(y_hat, self.training['ydata'][p][j * minibatch:(j + 1) * minibatch], eta)
                if bool(self.validation) is False:
                    acc = self.accuracy(self.test['xdata'], self.test['ydata'])
                    print(string + "test acc: %lf " % (acc), end='')
                else:
                    v_acc = self.accuracy(self.validation['xdata'], self.validation['ydata'])
                    print(string + "valid acc: %lf " % (v_acc), end='')
            t_acc = self.accuracy(self.training['xdata'], self.training['ydata'])
            self.train_acc.append(t_acc)
            if bool(self.validation) is False:
                acc = self.accuracy(self.test['xdata'], self.test['ydata'])
                self.test_acc.append(acc)
                print("[epoch end] train acc: %lf, test acc: %lf " % (t_acc, acc), end='')
            else:
                v_acc = self.accuracy(self.validation['xdata'], self.validation['ydata'])
                self.valid_acc.append(v_acc)
                print("[epoch end] train acc: %lf, valid acc: %lf " % (t_acc, v_acc), end='')
            if decay is True and (i + 1) % int(np.ceil(epoch / 3)) == 0:
                eta /= 2.0
                self.decay_pts.append(i)
                print("Learning rate decay by 2: %lf" % (eta), end='')
            print('')
    
    # training method
    # fake keras interface
    def fit(self, train_x, train_y, config, valid=None, test_x=None, test_y=None):
        self.config = config
        self.training = {}
        self.training['xdata'] = train_x
        self.training['ydata'] = train_y
        training_size = self.training['xdata'].shape[0]
        self.validation = {}
        self.test = {}
        if valid is not None and valid > 0.0 and valid < 1.0:
            self.training['xdata'], self.validation['xdata'], _ = np.split(self.training['xdata'], [int((1 - valid) * training_size), training_size])
            self.training['ydata'], self.validation['ydata'], _ = np.split(self.training['ydata'], [int((1 - valid) * training_size), training_size])
            validation_size = int(valid * training_size)
            training_size = int((1 - valid) * training_size)
            print(f'Train on {training_size} samples, validate on {validation_size} samples')
        elif test_x is not None and test_y is not None:
            self.test['xdata'] = test_x
            self.test['ydata'] = test_y
            test_size = self.test['xdata'].shape[0]
            print(f'Train on {training_size} samples, test on {test_size} samples')
        else:
            assert False, "need validation set or test set"
        if config['optimizer'] == 'sgd':
            self.sgd(config['eta'], config['epoch'],
                     config['minibatch'], config['momentum'],
                     config['decay'])
        else:
            # TODO
            # other optimizers
            pass
    
    # save model and training process
    def save(self, filename):
        with h5py.File(filename, 'w') as hf:
            for n in self.net:
                hf.create_dataset(n + '_w', data=self.net[n].weight)
                hf.create_dataset(n + '_b', data=self.net[n].bias)
            hf.create_dataset('train_acc', data=self.train_acc)
            hf.create_dataset('valid_acc', data=self.valid_acc)
            hf.create_dataset('test_acc', data=self.test_acc)
            hf.create_dataset('decay_pts', data=self.decay_pts)

    # load model and training process
    def load(self, filename):
        with h5py.File(filename, 'r') as hf:
            for n in self.net:
                self.net[n].set_par(hf[n + '_w'][:], hf[n + '_b'][:])
            self.train_acc = hf['train_acc'][:]
            self.valid_acc = hf['valid_acc'][:]
            self.test_acc = hf['test_acc'][:]
            self.decay_pts = hf['decay_pts'][:]

    # plot the learning curve
    def plot(self):
        title = 'MLP-'
        for i, l in enumerate(self.arch['layer']):
            title += str(self.arch['layer'][l]['num'])
            if i != 0:
                title += '(' + self.arch['layer'][l]['activation'] + ')'
            if i != len(self.arch['layer'].keys()) - 1:
                title += '-'
        title += ',epoch=' + str(self.config['epoch'])
        title += ',minibatch=' + str(self.config['minibatch'])
        title += ',eta=' + str(self.config['eta'])
        if self.config['decay'] == True:
            title += '(decay)'
        if self.arch['regularizer'] is not None:
            title += ',' + self.arch['regularizer'][0] + '=' + str(self.arch['regularizer'][1])
        plt.title(title)
        plt.xlabel("epoch(s)")
        plt.ylabel("accuracy")
        plt.plot(self.train_acc, label='train accuracy')
        if bool(self.validation) is False:
            plt.plot(self.test_acc, label='test accuracy')
        else:
            plt.plot(self.valid_acc, label='validation accuracy')
        for i, pt in enumerate(self.decay_pts):
            if i == 0:
                plt.scatter(pt, self.train_acc[pt], color='red', label="decay points")
            else:
                plt.scatter(pt, self.train_acc[pt], color='red')
            if bool(self.validation) is False:
                plt.scatter(pt, self.test_acc[pt], color='red')
            else:
                plt.scatter(pt, self.valid_acc[pt], color='red')
        plt.legend()
        plt.show()
