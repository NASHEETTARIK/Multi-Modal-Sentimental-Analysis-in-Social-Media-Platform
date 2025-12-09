import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Concatenate, Conv1D, Softmax, Reshape
from tensorflow.keras.models import Model


"""========================================================================
         Densely connected recurrent network with dual Attention    
   ========================================================================"""  


# Define the SelfAttention Layer
class SelfAttention(Layer):
    def __init__(self, h, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.h = h
        self.local_conv = Conv1D(filters=h, kernel_size=1, padding='same', use_bias=False)
        self.global_conv = Conv1D(filters=h, kernel_size=3, padding='same', use_bias=False)
        self.softmax = Softmax()

    def call(self, X):
        f_X = self.local_conv(X)  # Shape: (batch_size, time_steps, h)
        g_X = self.global_conv(X)  # Shape: (batch_size, time_steps, h)

        # Compute attention scores
        M = tf.matmul(f_X, g_X, transpose_b=True)  # Shape: (batch_size, time_steps, time_steps)
        M = M / tf.sqrt(tf.cast(self.h, tf.float32))  # Scale by sqrt(h)
        w = self.softmax(M)  # Attention weights

        # Apply attention weights
        attention_output = tf.matmul(w, X)  # Shape: (batch_size, time_steps, h)

        return attention_output

# Define the CrossPositionalAttention Layer
class CrossPositionalAttention(Layer):
    def __init__(self, **kwargs):
        super(CrossPositionalAttention, self).__init__(**kwargs)
        self.conv1x1_M = Conv1D(filters=64, kernel_size=1, padding='same', use_bias=False)
        self.conv1x1_N = Conv1D(filters=64, kernel_size=1, padding='same', use_bias=False)
        self.conv1x1_V = Conv1D(filters=64, kernel_size=1, padding='same', use_bias=False)
        self.softmax = Softmax()

    def call(self, F):
        M = self.conv1x1_M(F)  # Shape: (batch_size, time_steps, C)
        N = self.conv1x1_N(F)  # Shape: (batch_size, time_steps, C)
        V = self.conv1x1_V(F)  # Shape: (batch_size, time_steps, C)

        # Compute attention map
        Qx = M
        Dx = tf.transpose(N, perm=[0, 2, 1])  # Transpose N for dot product
        correlation = tf.matmul(Qx, Dx)  # Shape: (batch_size, time_steps, time_steps)
        S_z = self.softmax(correlation)  # Shape: (batch_size, time_steps, time_steps)

        # Update feature map
        new_feature_map = tf.matmul(S_z, V)  # Shape: (batch_size, time_steps, C)
        Y = new_feature_map + F  # Add original feature map F

        return Y

# Define the DCRN with Dual Attention
def build_dcrn_with_dual_attention(input_shape, lstm_units=64, attention_units=64):
    # Input layer (reshaped to add the time dimension)
    inputs = Input(shape=input_shape)
    reshaped_inputs = Reshape((input_shape[0], 1))(inputs)  # Reshape (500,) to (500, 1)
    # Self-Attention
    self_attention = SelfAttention(h=attention_units)(reshaped_inputs)
    # Cross-Positional Attention
    cross_positional_attention = CrossPositionalAttention()(self_attention)
    # LSTM Layers with dense connections
    x1 = LSTM(lstm_units, return_sequences=True)(cross_positional_attention)
    x2 = LSTM(lstm_units, return_sequences=True)(Concatenate()([cross_positional_attention, x1]))
    x3 = LSTM(lstm_units, return_sequences=False)(Concatenate()([cross_positional_attention, x1, x2]))
    # Dense layer
    dense_output = Dense(32, activation='tanh')(x3)
    outputs = Dense(6, activation='softmax')(dense_output)
    model = Model(inputs, outputs)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Train_Audio(X_train_audio,y_train):

    
    # Define input shape for your data (500 features)
    input_shape = (X_train_audio.shape[1],)
    
    # Build and summarize the model
    model = build_dcrn_with_dual_attention(input_shape=input_shape)
    # model.summary()
    
    model.fit(X_train_audio, y_train,
                        validation_data=(X_train_audio, y_train),
                        epochs=300,
                        batch_size=32)
    return model

def Train_Audio1(X_train,y_train):

    
    # Define input shape for your data (500 features)
    input_shape = (X_train.shape[1],)
    
    # Build and summarize the model
    model = build_dcrn_with_dual_attention(input_shape=input_shape)
    # model.summary()
    
    model.fit(X_train, y_train,
                        validation_data=(X_train, y_train),
                        epochs=300,
                        batch_size=32)
    return model




from tensorflow.keras.models import load_model

def Test_Audio(X_test):

    name = 'Features/model_Audio.h5'
    model = load_model(name, custom_objects={'SelfAttention': SelfAttention,
                                                    'CrossPositionalAttention': CrossPositionalAttention})
    pred = model.predict(X_test)
    return pred

def Test_Audio1(X_test):

    name = 'Features1/model_Audio.h5'
    model = load_model(name, custom_objects={'SelfAttention': SelfAttention,
                                                    'CrossPositionalAttention': CrossPositionalAttention})
    pred = model.predict(X_test)
    return pred






#  Densely connected recurrent network with dual Attention   

# inputs -> dual Attention (self_attention -> cross_positional_attention) -> recurrent network(LSTM) -> outputs(activation='softmax') ->compile(optimizer='adam', loss='categorical_crossentropy')


#%%%

# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_3 (InputLayer)           [(None, 509)]        0           []                               
                                                                                                  
#  reshape (Reshape)              (None, 509, 1)       0           ['input_3[0][0]']                
                                                                                                  
#  self_attention (SelfAttention)  (None, 509, 1)      256         ['reshape[0][0]']                
                                                                                                  
#  cross_positional_attention (Cr  (None, 509, 64)     192         ['self_attention[0][0]']         
#  ossPositionalAttention)                                                                          
                                                                                                  
#  lstm (LSTM)                    (None, 509, 64)      33024       ['cross_positional_attention[0][0
#                                                                  ]']                              
                                                                                                  
#  concatenate_1 (Concatenate)    (None, 509, 128)     0           ['cross_positional_attention[0][0
#                                                                  ]',                              
#                                                                   'lstm[0][0]']                   
                                                                                                  
#  lstm_1 (LSTM)                  (None, 509, 64)      49408       ['concatenate_1[0][0]']          
                                                                                                  
#  concatenate_2 (Concatenate)    (None, 509, 192)     0           ['cross_positional_attention[0][0
#                                                                  ]',                              
#                                                                   'lstm[0][0]',                   
#                                                                   'lstm_1[0][0]']                 
                                                                                                  
#  lstm_2 (LSTM)                  (None, 64)           65792       ['concatenate_2[0][0]']          
                                                                                                  
#  dense_12 (Dense)               (None, 32)           2080        ['lstm_2[0][0]']                 
                                                                                                  
#  dense_13 (Dense)               (None, 6)            198         ['dense_12[0][0]']               
                                                                                                  
# ==================================================================================================
# Total params: 150,950
# Trainable params: 150,950
# Non-trainable params: 0
# __________________________________________________________________________________________________
