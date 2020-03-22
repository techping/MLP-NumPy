# Multi-layer Perceptron
# Main Program
# Ziping Chen
# March 2020

from MLP import MLP
from config import net_conf
from data import get_data

# get training data
x_train, y_train, x_test, y_test = get_data('mnist')

# define model
model = MLP(net_conf['architecture'])
# print model summary
model.summary()
# (1) fit model with split validation
model.fit(x_train,
        y_train,
        net_conf['training'],
        valid=1/6)
# (2) fit model and validate with test set
# model.fit(x_train,
#         y_train,
#         net_conf['training'],
#         test_x=x_test,
#         test_y=y_test)
# plot learning curve with accuracy
model.plot()
# evaluate dataset
# model.evaluate(your_dataset)
# save model
# model.save('your_path.h5py')