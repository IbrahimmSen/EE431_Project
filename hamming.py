import numpy as np

G = np.array([
    [1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0]
], dtype=int)

H = np.array([
    [1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 1]
], dtype=int)

N_HAMMING = 7
K_HAMMING = 4

def hamming_encode(message_bits):
    u = np.array(message_bits, dtype=int)
    if u.shape[0] != K_HAMMING:
        raise ValueError(f"Error: Message block must be exactly {K_HAMMING} bits!")
    raw_multiplication = np.dot(u, G)
    codeword = raw_multiplication % 2
    return codeword

def hamming_decode(received_bits):
    r = np.array(received_bits, dtype=int)
    
    if r.shape[0] != N_HAMMING:
        raise ValueError(f"Error: Received block must be exactly {N_HAMMING} bits!")
        
    H_transpose = H.T
    raw_syndrome = np.dot(r, H_transpose)
    syndrome = raw_syndrome % 2
    
    if np.all(syndrome == 0):
        return r[0:K_HAMMING]
        
    error_index = -1
    for i in range(N_HAMMING):
        if np.array_equal(H[:, i], syndrome):
            error_index = i
            break
            
    if error_index != -1:
        r[error_index] = 1 - r[error_index]
        
    return r[0:K_HAMMING]