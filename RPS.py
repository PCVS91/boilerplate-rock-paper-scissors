def player(prev_play, opponent_history=[]):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.utils import to_categorical

    opponent_history.append(prev_play)
    opponent_history_encoded = []
    training_length = 12
    current_game = len(opponent_history)
    def encode(move_str):
        moves_dict = {'R': 0, 'P': 1, 'S': 2}
        if move_str in moves_dict:
            return moves_dict[move_str]
        else:
            raise ValueError(f"Invalid move: {move_str}")

    def decode(move_int):
        moves = ['R', 'P', 'S']
        return moves[move_int]

    # Initialize player history
    if not hasattr(player, 'player_history'):
        player.player_history = [decode(np.random.randint(0, 3)) for _ in range(training_length)]

    # Build model
    if not hasattr(player, 'model') or player.model is None:
        player.model = keras.Sequential([
            keras.layers.Dense(units=22, activation='relu', input_shape=(22,)),
            keras.layers.Dense(units=10, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=5, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=3, activation='softmax')
        ])
        player.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    for move in opponent_history:
        if move:
            opponent_history_encoded.append(encode(move))

    player_history_encoded = [encode(move) for move in player.player_history]

    if current_game < training_length:
        return decode(np.random.randint(0, 3))

    # Train and predict
    two_last_opponent = opponent_history_encoded[-3] * 3 + opponent_history_encoded[-2]
    two_last_player = player_history_encoded[-3] * 3 + player_history_encoded[-2]

    x_train = np.array([opponent_history_encoded[-11:-1] + player_history_encoded[-11:-1]])
    x_train = np.append(x_train, [two_last_opponent, two_last_player]).reshape(1, -1)
    y_train = to_categorical([opponent_history_encoded[-1]], num_classes=3)

    prediction = np.argmax(player.model.predict(x_train, verbose=0), axis=1)[0]
    failed = prediction != opponent_history_encoded[-1]

    player.model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)

    if failed and np.random.random() < 0.5:
        random_guess = decode(np.random.randint(0, 3))
        player.player_history.append(random_guess)
        if len(player.player_history) > training_length:
            player.player_history.pop(0)
        if current_game == 1000:
            opponent_history.clear()
            player.player_history = [decode(np.random.randint(0, 3)) for _ in range(training_length)]
            player.model = None
        return random_guess

    two_last_opponent_predict = opponent_history_encoded[-2] * 3 + opponent_history_encoded[-1]
    two_last_player_predict = player_history_encoded[-2] * 3 + player_history_encoded[-1]

    prediction_input = np.array([opponent_history_encoded[-10:] + player_history_encoded[-10:]])
    prediction_input = np.append(prediction_input, [two_last_opponent_predict, two_last_player_predict]).reshape(1, -1)
    prediction = player.model.predict(prediction_input, verbose=0)
    predicted_move = np.argmax(prediction, axis=1)[0]

    player.player_history.append(decode((predicted_move + 1) % 3))
    if len(player.player_history) > training_length:
        player.player_history.pop(0)

    if current_game == 1000:
        opponent_history.clear()
        player.player_history = [decode(np.random.randint(0, 3)) for _ in range(training_length)]
        player.model = None

    if np.random.random() < 0.1:
        return decode(np.random.randint(0, 3))

    return decode((predicted_move + 1) % 3)
