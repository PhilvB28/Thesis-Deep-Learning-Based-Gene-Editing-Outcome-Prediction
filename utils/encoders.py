import torch
import torch.nn.functional as F
import itertools

def find_microhomologies(sequence):
    """
    Finds microhomologies around a CRISPR cut site in a 60nt DNA sequence.

    Args:
        sequence (str): A 60nt DNA sequence where the cut site is at position 30.

    Returns:
        list: A list of microhomologous sequences.
    """

    if len(sequence) != 60:
        print(f"FEHLER in find_microhomologies: Länge = {len(sequence)}, Sequenz = {sequence}")
        raise ValueError("The input DNA sequence must be exactly 60 nucleotides long.")

    # Initialize variables
    cut_site = 30
    left = sequence[:cut_site]  # (positions 0 to 29)
    right = sequence[cut_site:]  # (positions 30 to 59)
    microhomologies = []

    # Find microhomologies
    max_length = min(len(left), len(right))  # Maximum length for microhomologies
    for i in range(max_length):
        if left[-(i + 1):] == right[:i + 1]:  # Check for matching sequences
            microhomologies.append(left[-(i + 1):])

    return microhomologies

#One Hot encoding including GC frac and MH-count
def dna_to_onehot_extra(dna_sequences, sequence_length=60, device=None):
    # Set device to cuda if available (or use provided device)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = len(dna_sequences)
    num_channels = 6  # A, T, C, G, GC content, Stem Loop Count

    # Initialize a zero tensor on the desired device
    onehot_matrix = torch.zeros((batch_size, num_channels, sequence_length),
                                dtype=torch.float32, device=device)

    # Define the mapping from nucleotide to channel index
    one_mer_comb_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    for batch_index, dna_sequence in enumerate(dna_sequences):
        # Truncate or pad the sequence to the desired length
        if len(dna_sequence) > sequence_length:
            dna_sequence = dna_sequence[:sequence_length]
        elif len(dna_sequence) < sequence_length:
            dna_sequence = dna_sequence.ljust(sequence_length, 'A')  # Pad with 'A'

        # Convert the DNA string to indices (defaulting invalid nucleotides to 0)
        indices = [one_mer_comb_map.get(nuc, 0) for nuc in dna_sequence]
        # Create a tensor of indices directly on the target device
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)

        # One-hot encode the indices (resulting shape: [sequence_length, 4])
        onehot_nucleotides = F.one_hot(indices_tensor, num_classes=4).float()
        # Transpose to shape: [4, sequence_length]
        onehot_nucleotides = onehot_nucleotides.transpose(0, 1)

        # Assign the one-hot encoded nucleotide channels to the first 4 channels
        onehot_matrix[batch_index, :4, :] = onehot_nucleotides

        # Calculate GC content for the sequence (uniform across the sequence length)
        gc_count = sum(1 for base in dna_sequence if base in 'GC')
        gc_content = gc_count / len(dna_sequence)
        onehot_matrix[batch_index, 4, :] = gc_content

        # Calculate microhomology count (assuming find_microhomologies is defined)
        microhomologies = find_microhomologies(dna_sequence)
        mh_count = len(microhomologies)
        onehot_matrix[batch_index, 5, :] = mh_count

    return onehot_matrix.squeeze(0)


#Simple one Hot encoding
def dna_to_onehot(dna_sequences, sequence_length=60, device=None):
    # Set device to CUDA if available (or use provided device)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = len(dna_sequences)
    num_channels = 4  # A, T, C, G

    # Initialize a zero tensor on the target device
    onehot_matrix = torch.zeros((batch_size, num_channels, sequence_length),
                                dtype=torch.float32, device=device)

    # Define the mapping from nucleotide to index
    one_mer_comb_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # Convert DNA sequences to index tensors
    indices_list = []
    for dna_sequence in dna_sequences:
        # Truncate or pad sequence
        dna_sequence = (dna_sequence[:sequence_length] + 'A' * sequence_length)[:sequence_length]

        # Convert characters to indices, defaulting invalid nucleotides to 0 (A)
        indices = [one_mer_comb_map.get(nuc, 0) for nuc in dna_sequence]
        indices_list.append(indices)

    # Create a batch tensor of indices on the device
    indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=device)

    # One-hot encode the indices
    onehot_nucleotides = F.one_hot(indices_tensor, num_classes=num_channels).float()

    # Transpose to shape: [batch_size, num_channels, sequence_length]
    onehot_matrix = onehot_nucleotides.permute(0, 2, 1)

    return onehot_matrix.squeeze(0)


#k-mer one-hot encoding
def dna_to_kmer_onehot(dna_sequences, k=2, sequence_length=60, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nucleotides = ['A', 'T', 'C', 'G']
    num_kmers = len(nucleotides) ** k
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    kmer_map = {mer: idx for idx, mer in enumerate(all_kmers)}

    batch_size = len(dna_sequences)
    effective_length = sequence_length - k + 1  # number of kmers in a sequence of length L
    if effective_length < 1:
        raise ValueError("sequence_length must be >= k")

    indices_list = []
    for dna_sequence in dna_sequences:
        # Truncate or pad the sequence
        dna_sequence = (dna_sequence[:sequence_length] + 'A' * sequence_length)[:sequence_length]

        # Extract overlapping k-mers
        kmers = [dna_sequence[i:i+k] for i in range(len(dna_sequence) - k + 1)]
        indices = [kmer_map.get(kmer, 0) for kmer in kmers]  # default to first k-mer (e.g. "A"*k)
        indices_list.append(indices)

    indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=device)
    onehot_kmers = F.one_hot(indices_tensor, num_classes=num_kmers).float()
    onehot_matrix = onehot_kmers.permute(0, 2, 1)

    return onehot_matrix.squeeze(0)
