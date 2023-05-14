# coding=utf-8
#  citing: https://github.com/jsyoon0823/GAIN
import sys
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.preprocess.utils import normalization, renormalization, rounding
from src.preprocess.utils import xavier_init
from src.preprocess.utils import binary_sampler, uniform_sampler, sample_batch_index

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)



def generator(x, m, theta_G):
    # Concatenate Mask and Data
    G_W1, G_W2, G_W3, G_b1, G_b2, G_b3 = theta_G
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob



def train_gain(data_x, gain_parameters):
  graph = tf.Graph()
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  pd_norm_parameters = pd.DataFrame(norm_parameters)
  pd_norm_parameters.to_csv('data/analysis_data/norm_parameters.csv', index = False)
  
  norm_data_x = np.nan_to_num(norm_data)


  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)

  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)

  # Discriminator
  D_prob = discriminator(Hat_X, H)
  # GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session(config = config)
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations

  saver = tf.train.Saver([G_b1, G_b2, G_b3, G_W1, G_W2, G_W3])
  step = 0
  for it in tqdm(range(iterations)):    
    step += 1  
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
    if step % 2000 == 0:
        saver.save(sess, 'model/gain/gain.ckpt', global_step = step)



            

def impute_data(data_x):
    graph = tf.Graph()
    # Define mask matrix
    data_m = 1-np.isnan(data_x)

    # System parameters

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)


    #  # Normalization
    #  load parameters
    parameters = pd.read_csv('data/analysis_data/norm_parameters.csv')
    parametras = parameters.to_numpy()
    norm_data, norm_parameters = normalization(data_x, parameters)
    norm_data_x = np.nan_to_num(norm_data)

    #  load generator
    sess = tf.Session(config = config)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  
  
    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))


    #  saver = tf.train.import_meta_graph('model/gain.ckpt')
    saver = tf.train.Saver([G_b1, G_b2, G_b3, G_W1, G_W2, G_W3])
    #  saver.restore(sess, tf.train.latest_checkpoint('model'))

    saver.restore(sess, tf.train.latest_checkpoint('model/gain'))



    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(x,m):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob



  
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [None, dim])

    G_sample = generator(X, M)
            
    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, no, dim) 
    M_mb = data_m
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)  

    # Rounding
    imputed_data = rounding(imputed_data, data_x)  
          
    return imputed_data
    
