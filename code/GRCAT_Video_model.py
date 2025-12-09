import tensorflow as tf
from tensorflow.keras import layers, models

"""========================================================================
       Gated attention enclosed Residual context aware transformer  
   ========================================================================"""  


# Gated Attention Layer
class GatedAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        self.gate_dense = layers.Dense(units)
        self.output_dense = layers.Dense(units)

    def call(self, inputs):
        queries, keys, values = inputs
        queries_transformed = self.query_dense(queries)
        keys_transformed = self.key_dense(keys)
        values_transformed = self.value_dense(values)

        scores = tf.matmul(queries_transformed, keys_transformed, transpose_b=True)
        attention_weights = tf.nn.softmax(scores)

        context = tf.matmul(attention_weights, values_transformed)
        gate = tf.sigmoid(self.gate_dense(queries))
        output = context * gate
        return self.output_dense(output)


# Residual Layer
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.dense = layers.Dense(units)
        self.add = layers.Add()

    def build(self, input_shape):
        self.shortcut = layers.Dense(input_shape[-1])

    def call(self, inputs):
        x = self.dense(inputs)
        shortcut = self.shortcut(inputs)
        return self.add([x, shortcut])


# Context-Aware Transformer Layer
class ContextAwareTransformer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(ContextAwareTransformer, self).__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Build the Full Model
def build_model(input_shape_video1, input_shape_video2, num_classes):
    input_video1 = layers.Input(shape=input_shape_video1)
    input_video2 = layers.Input(shape=input_shape_video2)

    # Branch 1: Video 1 processing
    x1 = layers.Dense(512, activation='tanh')(input_video1)
    x1 = ResidualLayer(512)(x1)
    x1 = layers.Flatten()(x1)

    # Branch 2: Video 2 processing
    x2 = layers.Dense(256, activation='tanh')(input_video2)

    # Ensure x2 has correct shape for transformer
    x2 = tf.expand_dims(x2, axis=1)
    x2 = GatedAttentionLayer(256)([x2, x2, x2]) 
    # Transformer layer
    transformer = ContextAwareTransformer(embed_dim=256, num_heads=4, ff_dim=512)
    x2 = transformer(x2)
    x2 = tf.squeeze(x2, axis=1)
    x2 = layers.Flatten()(x2)

    # Combine branches
    combined = layers.Concatenate()([x1, x2])

    # Final fully connected layers
    combined = layers.Dense(512, activation='tanh')(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = models.Model(inputs=[input_video1, input_video2], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def Train_Video(X_train_video1,X_train_video2,y_train):

    input_shape_video1 = (X_train_video1.shape[1],)  # Shape of first input
    input_shape_video2 = (X_train_video2.shape[1],)  # Shape of second input
    
    num_classes = 6  # Example number of classes for classification
    
    # Build and summarize the model
    model = build_model(input_shape_video1, input_shape_video2, num_classes)
    model.summary()
    
    
    model.fit([X_train_video1, X_train_video2], y_train, epochs=300, batch_size=32,validation_data=([X_train_video1, X_train_video2],y_train))
    
    return model



def Train_Video1(X_train_video1,X_train_video2,y_train):

    input_shape_video1 = (X_train_video1.shape[1],)  # Shape of first input
    input_shape_video2 = (X_train_video2.shape[1],)  # Shape of second input
    
    num_classes = 6  # Example number of classes for classification
    
    # Build and summarize the model
    model = build_model(input_shape_video1, input_shape_video2, num_classes)
    # model.summary()
    
    
    model.fit([X_train_video1, X_train_video2], y_train, epochs=300, batch_size=32,validation_data=([X_train_video1, X_train_video2],y_train))
    
    return model



from tensorflow.keras.models import load_model



def Test_Video(X_test1,X_test2):

    name = 'Features/model_Video.h5'
    model = load_model(name, custom_objects={'GatedAttentionLayer': GatedAttentionLayer,
                                                    'ResidualLayer': ResidualLayer,'ContextAwareTransformer':ContextAwareTransformer})
    pred = model.predict([X_test1,X_test2])
    return pred


def Test_Video1(X_test1,X_test2):

    name = 'Features1/model_Video.h5'
    model = load_model(name, custom_objects={'GatedAttentionLayer': GatedAttentionLayer,
                                                    'ResidualLayer': ResidualLayer,'ContextAwareTransformer':ContextAwareTransformer})
    pred = model.predict([X_test1,X_test2])
    return pred





#%%%
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 1864)]       0           []                               
                                                                                                  
#  dense_2 (Dense)                (None, 256)          477440      ['input_2[0][0]']                
                                                                                                  
#  tf.expand_dims (TFOpLambda)    (None, 1, 256)       0           ['dense_2[0][0]']                
                                                                                                  
#  input_1 (InputLayer)           [(None, 5297)]       0           []                               
                                                                                                  
#  gated_attention_layer (GatedAt  (None, 1, 256)      328960      ['tf.expand_dims[0][0]',         
#  tentionLayer)                                                    'tf.expand_dims[0][0]',         
#                                                                   'tf.expand_dims[0][0]']         
                                                                                                  
#  dense (Dense)                  (None, 512)          2712576     ['input_1[0][0]']                
                                                                                                  
#  context_aware_transformer (Con  (None, 1, 256)      1315840     ['gated_attention_layer[0][0]']  
#  textAwareTransformer)                                                                            
                                                                                                  
#  residual_layer (ResidualLayer)  (None, 512)         525312      ['dense[0][0]']                  
                                                                                                  
#  tf.compat.v1.squeeze (TFOpLamb  (None, 256)         0           ['context_aware_transformer[0][0]
#  da)                                                             ']                               
                                                                                                  
#  flatten (Flatten)              (None, 512)          0           ['residual_layer[0][0]']         
                                                                                                  
#  flatten_1 (Flatten)            (None, 256)          0           ['tf.compat.v1.squeeze[0][0]']   
                                                                                                  
#  concatenate (Concatenate)      (None, 768)          0           ['flatten[0][0]',                
#                                                                   'flatten_1[0][0]']              
                                                                                                  
#  dense_10 (Dense)               (None, 512)          393728      ['concatenate[0][0]']            
                                                                                                  
#  dropout_2 (Dropout)            (None, 512)          0           ['dense_10[0][0]']               
                                                                                                  
#  dense_11 (Dense)               (None, 6)            3078        ['dropout_2[0][0]']              
                                                                                                  
# ==================================================================================================
# Total params: 5,756,934
# Trainable params: 5,756,934
# Non-trainable params: 0
# __________________________________________________________________________________________________
