import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt



class ImprovedTAPCell(layers.Layer):
    """
    Proposition conceptuelle pour améliorer le modèle TAP (ArXiv 2102.05095).
    Ajout d'une mémoire externe pour les dépendances long terme.
    """
    def __init__(self, hidden_dim, memory_size, **kwargs):
        super(ImprovedTAPCell, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Mémoire Externe (ex: Neural Turing Machine simplified)
        self.memory_bank = self.add_weight(shape=(memory_size, hidden_dim),
                                           initializer="zeros",
                                           trainable=False) # Ou True si on apprend à écrire
                                           
        # Layers pour Query, Key, Value de l'attention sur la mémoire
        self.W_q = layers.Dense(hidden_dim)
        self.W_k = layers.Dense(hidden_dim)
        self.W_v = layers.Dense(hidden_dim)

    def call(self, current_latent_state):
        # 1. Calcul Standard TAP (transition temporelle)
        # next_state = tap_transition(current_latent_state) ...
        
        # 2. Mécanisme d'amélioration: Récupération depuis la mémoire (Read)
        query = self.W_q(current_latent_state)
        keys = self.W_k(self.memory_bank)
        values = self.W_v(self.memory_bank)
        
        # Scaled Dot Product Attention sur la mémoire
        d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
        scores = tf.matmul(query, keys, transpose_b=True) / tf.math.sqrt(d_k)
        weights = tf.nn.softmax(scores)
        
        memory_context = tf.matmul(weights, values)
        
        # 3. Fusion de l'état latent et du contexte mémoire
        # enhanced_state = next_state + memory_context (Residual connection)
        enhanced_state = current_latent_state + memory_context # Simplifié
        
        return enhanced_state