import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import src.data_engineering.data_engineering as de
import src.model.lstm as lstm


parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--path_scaler_dataset',  type=str,
                   help='patch to the folder where the batched dataset \
                   and the used scaler object is stored')
parser.add_argument('--N_EPOCH', type=int,
                   help='number of epochs to train')
parser.add_argument('--n_pred', type=int,
                   help='number of timesteps to forcast per timeseries')
parser.add_argument('--size_batch', type=int,
                   help='number of elements in each batch')
parser.add_argument('--n_feat', type=int,
                   help='number of features in the dataset')
parser.add_argument('--size_out', type=int,
                   help='size of output (target)')
parser.add_argument('--size_hidden', type=int,
                   help='size of hidden layers in the LSTM network')
args, unknown = parser.parse_known_args()

print('args', args)
print('unknown', unknown)


def sMAPE(y_true, y_pred):
    loss = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y_pred,y_true)),
                                        tf.divide(tf.abs(y_pred) + tf.abs(y_true),2.)
                                   )
                         )
    return loss

def loss_func(model, X, y, training=True):
    y_pred = model.forward(X,training=training)
    loss = sMAPE(y,y_pred)
    return loss

def gradients(model, X, y):
    with tf.GradientTape() as tape:
        loss = loss_func(model,X,y,training=True)
    return loss, tape.gradient(loss, model.trainable_variables)

def train_step(X_batch, y_batch, model, loss_avg, epoch, step):
    X_batch = tf.cast(X_batch ,tf.float32)
    y_batch = tf.cast(y_batch ,tf.float32)

    loss, grads = gradients(model, X_batch, y_batch)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_avg.update_state(loss)  # Add current batch loss

    print(f'Loss of step {step+1}, epoch {epoch+1}: {np.array(loss)}')



    
    
if __name__ == '__main__':
    
    
    
    
    train_dataset, scaler =  de.data_loader(args.path_scaler_dataset,
                 size_out=args.n_pred,
                 size_batch=args.size_batch,
                 norm_in=False,
                 norm_out=False,
                 target=True, 
                 impute=False,
                 scale='standard')

    model = lstm.LSTM_timeseries(args.n_feat, args.size_hidden, args.size_out, n_pred=args.n_pred)


    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
    #keep track of results per epoch
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(args.N_EPOCH):

        epoch_loss_avg = tf.keras.metrics.Mean()
        for i, (X_batch, y_batch) in enumerate(train_dataset):
            train_step(X_batch,y_batch,model=model,loss_avg=epoch_loss_avg,epoch=epoch,step=i)
        
        train_loss_results.append(epoch_loss_avg.result())

        
  
