import numpy as np
import matplotlib.pyplot as plt
from hamming import hamming_encode, hamming_decode
from convolutional import convolutional_encode, viterbi_decode
from channel import binary_symmetric_channel, calculate_bit_errors

def run_hamming_only_pipeline(message_bits, ber, seed=None):
    hamming_encoded = []
    for i in range(0, len(message_bits), 4):
        block = message_bits[i:i+4]
        encoded_block = hamming_encode(block)
        hamming_encoded.extend(encoded_block)
        
    channel_output = binary_symmetric_channel(hamming_encoded, ber, seed=seed)
    
    decoded_bits = []
    for i in range(0, len(channel_output), 7):
        block = channel_output[i:i+7]
        decoded_block = hamming_decode(block)
        decoded_bits.extend(decoded_block)
        
    return hamming_encoded, channel_output, decoded_bits

def run_convolutional_only_pipeline(message_bits, ber, seed=None):
    fully_encoded = convolutional_encode(message_bits)
    channel_output = binary_symmetric_channel(fully_encoded, ber, seed=seed)
    decoded_bits = viterbi_decode(channel_output)
    return fully_encoded, channel_output, decoded_bits

def run_concatenated_pipeline(message_bits, ber, seed=None):
    hamming_encoded = []
    for i in range(0, len(message_bits), 4):
        block = message_bits[i:i+4]
        encoded_block = hamming_encode(block)
        hamming_encoded.extend(encoded_block)
        
    fully_encoded = convolutional_encode(hamming_encoded)
    channel_output = binary_symmetric_channel(fully_encoded, ber, seed=seed)
    viterbi_decoded = viterbi_decode(channel_output)
    
    final_decoded_bits = []
    for i in range(0, len(viterbi_decoded), 7):
        block = viterbi_decoded[i:i+7]
        decoded_block = hamming_decode(block)
        final_decoded_bits.extend(decoded_block)
        
    return fully_encoded, channel_output, final_decoded_bits

def main():
    personal_message = "ECC2026-S02B"
    seeds = [4129, 9533, 17021]
    channel_bers = [0.001, 0.01, 0.05, 0.10, 0.15]
    
    print("--- System Parameters ---")
    print(f"Message: {personal_message}")
    print(f"Seeds: {seeds}")
    print(f"BER List: {channel_bers}\n")
    
    message_bytes = personal_message.encode('utf-8')
    source_bits = []
    for b in message_bytes:
        bits_str = bin(b)[2:].zfill(8)
        source_bits.extend([int(x) for x in bits_str])
        
    hamming_ber_matrix = np.zeros((len(seeds), len(channel_bers)))
    conv_ber_matrix = np.zeros((len(seeds), len(channel_bers)))
    concat_ber_matrix = np.zeros((len(seeds), len(channel_bers)))
    
    print("\n--- Main Simulation Loop Started ---")
    for s_idx, s in enumerate(seeds):
        print(f"\n>> Collecting data for Seed {s}:")
        for b_idx, ber in enumerate(channel_bers):
            
            _, _, dec_hamming = run_hamming_only_pipeline(source_bits, ber, seed=s)
            hamming_ber_matrix[s_idx, b_idx] = calculate_bit_errors(source_bits, dec_hamming) / len(source_bits)
            
            _, _, dec_conv = run_convolutional_only_pipeline(source_bits, ber, seed=s)
            conv_ber_matrix[s_idx, b_idx] = calculate_bit_errors(source_bits, dec_conv) / len(source_bits)
            
            _, _, dec_concat = run_concatenated_pipeline(source_bits, ber, seed=s)
            concat_ber_matrix[s_idx, b_idx] = calculate_bit_errors(source_bits, dec_concat) / len(source_bits)
            
            print(f"   BER: {ber:<5} | Hamming: {hamming_ber_matrix[s_idx, b_idx]:.4f} | Conv: {conv_ber_matrix[s_idx, b_idx]:.4f} | Concat: {concat_ber_matrix[s_idx, b_idx]:.4f}")
            
    avg_hamming = np.mean(hamming_ber_matrix, axis=0)
    avg_conv = np.mean(conv_ber_matrix, axis=0)
    avg_concat = np.mean(concat_ber_matrix, axis=0)
    
    plt.figure(figsize=(10, 7))
    plt.loglog(channel_bers, channel_bers, 'k--', alpha=0.5, label='Uncoded Baseline (Channel BER)')
    plt.loglog(channel_bers, avg_hamming, 'g^--', label='Hamming (7,4) Only')
    plt.loglog(channel_bers, avg_conv, 'ro--', label='Convolutional Only (Viterbi)')
    plt.loglog(channel_bers, avg_concat, 'bs-', linewidth=2, label='Serially Concatenated System')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Theoretical Channel BER')
    plt.ylabel('Measured Post-Decoding BER')
    plt.title('EE413 Project: Error Correction Performance Comparison')
    plt.legend()
    
    plt.savefig('ber_performance_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()