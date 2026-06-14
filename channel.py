import numpy as np

def binary_symmetric_channel(bits, ber, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    transmitted_bits = np.array(bits, dtype=int)
    num_bits = len(transmitted_bits)
    
    error_mask = np.random.rand(num_bits) < ber
    received_bits = transmitted_bits ^ error_mask.astype(int)
    
    return received_bits

def calculate_bit_errors(original_bits, decoded_bits):
    orig = np.array(original_bits, dtype=int)
    dec = np.array(decoded_bits, dtype=int)
    
    if len(orig) != len(dec):
        raise ValueError("Error: Length of bit sequences to compare must be equal!")
        
    error_count = np.sum(orig != dec)
    return error_count