# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

def player(prev_play, opponent_history=[]):
    
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.utils import to_categorical
    
    np.random.seed(42)
    opponent_history.append(prev_play)
    opponent_history_encoded=[]
    training_length=12
    current_game=len(opponent_history)
    training_moves=np.random.randint(0,3,training_length)
    
    #Build model
    # Initialize model as a static variable inside the function
    if current_game==1:
        player.model = keras.Sequential([
            keras.layers.Dense(units=10, activation='sigmoid'),
            keras.layers.Dense(units=20, activation='sigmoid'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(units=5, activation='sigmoid'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(units=3, activation='softmax')
        ])
        player.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    
    #%%


    def encode(move_str):
        moves_dict = {'R': 0, 'P': 1, 'S': 2}
        if move_str in moves_dict:
            return moves_dict[move_str]
        else:
            raise ValueError(f"Invalid move: {move_str}")
        
    def decode(move_int):
        moves= ['R', 'P', 'S']
        return moves[move_int]
    
    for move in opponent_history:
        if move:
            opponent_history_encoded.append(encode(move))
        
    
    if current_game<training_length:
        return decode(training_moves[current_game])
    
        
    #%% Train and predict
    x_train = np.array([opponent_history_encoded[-11:-1]]) 
    y_train = to_categorical([opponent_history_encoded[-1]], num_classes=3)
    player.model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
    
    prediction = player.model.predict(np.array([opponent_history_encoded[-10:]]),verbose=0)
    predicted_move = np.argmax(prediction, axis=1)[0]
    
    
    return decode((predicted_move+1)%3)
        
        
        
        
    #%%
    


