import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Вычисляем attention scores: decoder_hidden_state^T * W_mult * encoder_hidden_states
    attention_scores = decoder_hidden_state.T @ W_mult @ encoder_hidden_states
    
    # Применяем softmax к attention scores
    softmax_scores = softmax(attention_scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    attention_vector = encoder_hidden_states @ softmax_scores.T
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    n_states = encoder_hidden_states.shape[1]
    
    # Преобразуем состояния энкодера
    enc_transformed = W_add_enc @ encoder_hidden_states
    
    # Преобразуем состояние декодера
    dec_transformed = W_add_dec @ decoder_hidden_state
    
    # Расширяем преобразованное состояние декодера для суммирования
    dec_transformed_expanded = np.tile(dec_transformed, (1, n_states))
    
    # Вычисляем attention scores
    attention_scores = v_add.T @ np.tanh(enc_transformed + dec_transformed_expanded)
    
    # Применяем softmax к attention scores
    softmax_scores = softmax(attention_scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    attention_vector = encoder_hidden_states @ softmax_scores.T
    
    return attention_vector


out_dict = {
    'multiplicative_attention': multiplicative_attention,
    'additive_attention': additive_attention
}


np.save("submission_dict_hw08.npy", out_dict, allow_pickle=True)
print("File saved to `submission_dict_hw08.npy`")