import tensorflow as tf
from tensorflow.keras import layers, models



"""========================================================================
              Self Attention based Capsule Bi-lstm  
   ========================================================================"""   

# Define the squash activation function
def squash(vectors, axis=-1):
    norm = tf.norm(vectors, axis=axis, keepdims=True)
    return (norm**2 / (1 + norm**2)) * (vectors / norm)


# Self-Attention Dilated Convolution Block
def dilated_conv_block(input_layer, filters, dilation_rate):
    x = layers.Conv1D(filters, kernel_size=3, dilation_rate=dilation_rate, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    attention = layers.Add()([avg_pool, max_pool])
    attention = layers.Reshape((1, filters))(attention)
    attention = layers.Conv1D(filters, kernel_size=1, padding='same', activation='sigmoid')(attention)
    return layers.Multiply()([x, attention])



# Primary Capsule Layer
class PrimaryCaps(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule

    def build(self, input_shape):
        self.conv = layers.Conv1D(self.num_capsules * self.dim_capsule, kernel_size=3, strides=2, padding='same')

    def call(self, inputs):
        output = self.conv(inputs)
        output = layers.Reshape((-1, self.dim_capsule))(output)
        return squash(output)  # Use custom squash function




# Digital Capsule Layer
class DigitalCaps(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, **kwargs):
        super(DigitalCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule

    def build(self, input_shape):
        # Compute the expected output shape based on input shape
        input_dim = input_shape[-1]
        self.dense = layers.Dense(self.num_capsules * self.dim_capsule)

    def call(self, inputs):
        output = self.dense(inputs)
        # Reshape based on the number of capsules and the dimension of capsules
        output = layers.Reshape((-1, self.num_capsules, self.dim_capsule))(output)  # Adjust this line based on your needs
        return squash(output)  # Use custom squash function



# Model Architecture
def DACapsNet_BiLSTM(input_shape):
    
    inputs = layers.Input(shape=input_shape)
    x = dilated_conv_block(inputs, filters=64, dilation_rate=2)
    x = dilated_conv_block(x, filters=128, dilation_rate=4)
    primary_caps = PrimaryCaps(num_capsules=8, dim_capsule=16)(x)
    digital_caps = DigitalCaps(num_capsules=10, dim_capsule=16)(primary_caps)
    flat_caps = layers.Reshape(( digital_caps.shape[1], 10 * 16))(digital_caps)  # Reshape for LSTM (adjust based on actual output shape)
    bi_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(flat_caps)
    output = layers.Dense(6, activation='softmax')(bi_lstm)

    model = models(inputs,output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Train_Text(X_train_text,y_train):
    

    input_shape = (X_train_text.shape[1],1)
    
    # Build and summarize the model
    # Create model
    model = DACapsNet_BiLSTM(input_shape=input_shape)
    # model.summary()
    
    model.fit(X_train_text, y_train,
                        validation_data=(X_train_text, y_train),
                        epochs=300,
                        batch_size=32)

    return model



def Train_Text1(X_train,y_train):
    

    input_shape = (X_train.shape[1],1)
    
    # Build and summarize the model
    # Create model
    model = DACapsNet_BiLSTM(input_shape=input_shape)
    # model.summary()
    
    model.fit(X_train, y_train,
                        validation_data=(X_train, y_train),
                        epochs=300,
                        batch_size=32)

    return model



from tensorflow.keras.models import load_model


def Test_Text(X_test):

    name = 'Features/model_Text.h5'
    model = load_model(name, custom_objects={'PrimaryCaps': PrimaryCaps,
                                                    'DigitalCaps': DigitalCaps})
    pred = model.predict(X_test)
    return pred



def Test_Text1(X_test):

    name = 'Features1/model_Text.h5'
    model = load_model(name, custom_objects={'PrimaryCaps': PrimaryCaps,
                                                    'DigitalCaps': DigitalCaps})
    pred = model.predict(X_test)
    return pred



    
    
# inputs -> dilated_conv_block(Self-Attention Dilated Convolution Block) -> dilated_conv_block ->
# Capsule (primary_caps -> digital_caps ) -> bi_lstm -> output(activation='softmax') ->compile(optimizer='adam', loss='categorical_crossentropy')
    
#%%
   
    
# _______________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_5 (InputLayer)           [(None, 155, 1)]     0           []                               
                                                                                                  
#  conv1d_9 (Conv1D)              (None, 155, 64)      256         ['input_5[0][0]']                
                                                                                                  
#  batch_normalization_2 (BatchNo  (None, 155, 64)     256         ['conv1d_9[0][0]']               
#  rmalization)                                                                                     
                                                                                                  
#  re_lu_2 (ReLU)                 (None, 155, 64)      0           ['batch_normalization_2[0][0]']  
                                                                                                  
#  global_average_pooling1d_2 (Gl  (None, 64)          0           ['re_lu_2[0][0]']                
#  obalAveragePooling1D)                                                                            
                                                                                                  
#  global_max_pooling1d_2 (Global  (None, 64)          0           ['re_lu_2[0][0]']                
#  MaxPooling1D)                                                                                    
                                                                                                  
#  add_3 (Add)                    (None, 64)           0           ['global_average_pooling1d_2[0][0
#                                                                  ]',                              
#                                                                   'global_max_pooling1d_2[0][0]'] 
                                                                                                  
#  reshape_4 (Reshape)            (None, 1, 64)        0           ['add_3[0][0]']                  
                                                                                                  
#  conv1d_10 (Conv1D)             (None, 1, 64)        4160        ['reshape_4[0][0]']              
                                                                                                  
#  multiply_2 (Multiply)          (None, 155, 64)      0           ['re_lu_2[0][0]',                
#                                                                   'conv1d_10[0][0]']              
                                                                                                  
#  conv1d_11 (Conv1D)             (None, 155, 128)     24704       ['multiply_2[0][0]']             
                                                                                                  
#  batch_normalization_3 (BatchNo  (None, 155, 128)    512         ['conv1d_11[0][0]']              
#  rmalization)                                                                                     
                                                                                                  
#  re_lu_3 (ReLU)                 (None, 155, 128)     0           ['batch_normalization_3[0][0]']  
                                                                                                  
#  global_average_pooling1d_3 (Gl  (None, 128)         0           ['re_lu_3[0][0]']                
#  obalAveragePooling1D)                                                                            
                                                                                                  
#  global_max_pooling1d_3 (Global  (None, 128)         0           ['re_lu_3[0][0]']                
#  MaxPooling1D)                                                                                    
                                                                                                  
#  add_4 (Add)                    (None, 128)          0           ['global_average_pooling1d_3[0][0
#                                                                  ]',                              
#                                                                   'global_max_pooling1d_3[0][0]'] 
                                                                                                  
#  reshape_5 (Reshape)            (None, 1, 128)       0           ['add_4[0][0]']                  
                                                                                                  
#  conv1d_12 (Conv1D)             (None, 1, 128)       16512       ['reshape_5[0][0]']              
                                                                                                  
#  multiply_3 (Multiply)          (None, 155, 128)     0           ['re_lu_3[0][0]',                
#                                                                   'conv1d_12[0][0]']              
                                                                                                  
#  primary_caps_1 (PrimaryCaps)   (None, 624, 16)      49280       ['multiply_3[0][0]']             
                                                                                                  
#  digital_caps_1 (DigitalCaps)   (None, 624, 10, 16)  2720        ['primary_caps_1[0][0]']         
                                                                                                  
#  reshape_6 (Reshape)            (None, 624, 160)     0           ['digital_caps_1[0][0]']         
                                                                                                  
#  bidirectional_1 (Bidirectional  (None, 256)         295936      ['reshape_6[0][0]']              
#  )                                                                                                
                                                                                                  
#  dense_15 (Dense)               (None, 6)            1542        ['bidirectional_1[0][0]']        
                                                                                                  
# ==================================================================================================
# Total params: 395,878
# Trainable params: 395,878
# Non-trainable params: 0
# __________________________________________________________________________________________________    
    
    
