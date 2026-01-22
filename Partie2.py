import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def generate_synthetic_data(n_samples=1000, seq_len=50):
    """
    Génère des ondes sinusoïdales combinées pour simuler des séries temporelles.
    """
    X = []
    y = []
    for _ in range(n_samples):
        # Combinaison de fréquences aléatoires
        freq1 = np.random.uniform(0.05, 0.1)
        freq2 = np.random.uniform(0.01, 0.05)
        offset = np.random.uniform(0, 2*np.pi)
        
        t = np.linspace(0, 100, seq_len + 1)
        series = np.sin(2 * np.pi * freq1 * t + offset) + 0.5 * np.sin(2 * np.pi * freq2 * t)
        
        X.append(series[:-1]) # Input sequence
        y.append(series[-1])  # Next value prediction
        
    return np.expand_dims(np.array(X), -1), np.array(y)

# 1. Préparation des données
X, y = generate_synthetic_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Construction du Modèle Hybride (Bi-LSTM + Attention)
def build_hybrid_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    # Encoder: Bi-directional LSTM
    # return_sequences=True est crucial pour que l'attention puisse voir tous les états cachés
    lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    
    # Attention Layer (Custom implementation from Part 1)
    context_vector, attention_weights = SimpleAttention()(lstm_out)
    
    # Decoder / Prediction Head
    x = layers.Dense(32, activation="relu")(context_vector)
    outputs = layers.Dense(1)(x) # Prédiction de la prochaine valeur (Regression)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_LSTM_Attention")
    return model

model = build_hybrid_model(input_shape=(50, 1))
model.compile(optimizer="adam", loss="mse")
model.summary()

# 3. Entraînement
print("\nTraining Hybrid Model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# 4. Visualisation de l'Attention (Optionnel pour le rapport)
def visualize_attention(model, x_sample):
    # Créer un sous-modèle qui renvoie les poids d'attention
    att_model = keras.Model(inputs=model.input, 
                            outputs=model.get_layer("simple_attention").output)
    _, weights = att_model.predict(x_sample)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_sample[0, :, 0], label="Input Series")
    plt.plot(weights[0, :, 0], label="Attention Weights", color='red', linestyle='--')
    plt.title("Attention Weights overlay on Input Series")
    plt.legend()
    plt.show()

# Exemple de visualisation
visualize_attention(model, X_test[:1])