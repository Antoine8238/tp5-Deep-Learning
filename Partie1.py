import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class SimpleAttention(layers.Layer):
    
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape est (batch_size, seq_len, hidden_dim)
        # W : poids pour le calcul du score
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        # b : biais pour le calcul du score
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # 1. Calcul du score (Bahdanau style simplifié ou Dot product)
        # e = tanh(x . W + b)
        # x . W -> (batch, seq_len, 1)
        e = tf.tanh(tf.matmul(x, self.W) + self.b) 
        
        # 2. Calcul des poids d'alignement via Softmax
        # alignment_weights shape: (batch, seq_len, 1)
        alignment_weights = tf.nn.softmax(e, axis=1)
        
        # 3. Calcul du vecteur de contexte (somme pondérée)
        # context_vector = sum(x * weights)
        # Broadcasting weights to apply to hidden_dim
        context_vector = x * alignment_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, alignment_weights

# Test unitaire rapide de la couche
def test_attention_layer():
    print("Testing SimpleAttention Layer...")
    dummy_input = tf.random.normal((32, 10, 64)) # Batch 32, Seq 10, Dim 64
    attention_layer = SimpleAttention()
    context, weights = attention_layer(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Context shape: {context.shape}") # Devrait être (32, 64)
    print(f"Weights shape: {weights.shape}") # Devrait être (32, 10, 1)
    print("Test passed.\n")

test_attention_layer()