import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

def define_model(vocab_size=5000, embedding_dim=128, max_length=100):
    """
    Define and compile the LSTM-based model for product recommendation.
    
    Parameters:
    vocab_size (int): Size of the vocabulary.
    embedding_dim (int): Dimension of the embedding layer.
    max_length (int): Maximum length of the input sequences.
    
    Returns:
    tensorflow.keras.models.Sequential: Compiled LSTM-based model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Define model parameters
    vocab_size = 5000
    embedding_dim = 128
    max_length = 100
    
    # Define the model
    model = define_model(vocab_size, embedding_dim, max_length)
    
    # Print model summary
    model.summary()

if __name__ == "__main__":
    main()
