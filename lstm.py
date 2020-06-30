import tensorflow as tf

class LSTM_expsmooth(tf.Module):
    def __init__(self, nodes_in, units_hidden, nodes_out, n_pred=1):
        self.units_hidden = units_hidden
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.n_pred = n_pred
        
        self.W_i = tf.Variable(tf.zeros([self.nodes_in, self.units_hidden]),name='w_i')  
        self.U_i = tf.Variable(tf.zeros([self.units_hidden, self.units_hidden]),name='u_i')
        self.b_i = tf.Variable(tf.zeros([self.units_hidden]),name='b_i')
        self.W_f = tf.Variable(tf.zeros([self.nodes_in, self.units_hidden]),name='w_f')
        self.U_f = tf.Variable(tf.zeros([self.units_hidden, self.units_hidden]),name='u_f')
        self.b_f = tf.Variable(tf.zeros([self.units_hidden]),name='b_f')
        self.W_o = tf.Variable(tf.zeros([self.nodes_in, self.units_hidden]),name='w_o')
        self.U_o= tf.Variable(tf.zeros([self.units_hidden, self.units_hidden]),name='u_o')
        self.b_o = tf.Variable(tf.zeros([self.units_hidden]),name='b_o')
        self.W_c = tf.Variable(tf.zeros([self.nodes_in, self.units_hidden]),name='w_c')
        self.U_c = tf.Variable(tf.zeros([self.units_hidden, self.units_hidden]),name='u_c')
        self.b_c = tf.Variable(tf.zeros([self.units_hidden]),name='b_c')
        # output layer
        self.W_ol = tf.Variable(tf.random.truncated_normal([self.units_hidden, self.nodes_out], mean=0, stddev=.01),
                               name='w_ol')
        self.b_ol = tf.Variable(tf.random.truncated_normal([self.nodes_out], mean=0, stddev=.01),
                               name='b_ol')
       
    
    def forward_onestep(self, previous_hidden_memory_tuple, x):
        # runs one time step
        previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)

        i = tf.sigmoid( tf.matmul(x, self.W_i) +
                        tf.matmul(previous_hidden_state, self.U_i) + self.b_i)

        f = tf.sigmoid( tf.matmul(x, self.W_f) +
                        tf.matmul(previous_hidden_state, self.U_f) + self.b_f)

        o = tf.sigmoid( tf.matmul(x, self.W_o) +
                        tf.matmul(previous_hidden_state, self.U_o) + self.b_o)

        c_ = tf.nn.tanh(tf.matmul(x, self.W_c) +
                        tf.matmul(previous_hidden_state, self.U_c) + self.b_c)

        # Final Memory cell
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])
    
    def output(self, state_hidden):
            output = tf.nn.relu(tf.matmul(state_hidden, self.W_ol) + self.b_ol) #shape (size_seq, size_batch, nodes_out)
            return output
           
    def forward(self, X, training=True):
        
        self._inputs = X #shape (batch, size_seq, self.nodes_in)
        initial_hidden  = self._inputs[:, 0, :]
        initial_hidden = tf.matmul(initial_hidden,   
                                   tf.zeros([self.nodes_in, self.units_hidden], dtype=tf.dtypes.float32))
        self.initial_hidden_c = tf.stack([initial_hidden, initial_hidden])
        # transforming to shape (size_seq, size_batch, nodes_in) for scan function
        self.input_scan = tf.transpose(self._inputs, perm=[1, 0, 2])
        
        all_hidden_states = tf.scan(self.forward_onestep, self.input_scan, 
                                    initializer=self.initial_hidden_c, 
                                    name='states') #shape (size_seq, 2, size_batch, units_hidden)
        
        all_hidden_states = all_hidden_states[:, 0, :, :]  #shape (size_seq, size_batch, units_hidden)
        
        outputs = tf.map_fn(self.output, all_hidden_states) #shape (size_seq, size_batch, nodes_out)
        
        return tf.transpose(outputs[-self.n_pred:], perm=[1,0,2]) #just output of the last

