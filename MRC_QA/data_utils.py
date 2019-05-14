import numpy as np

def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_len = max([len(x) for x in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_tok] * max(max_len - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_len))
    return np.array(sequence_padded), np.array(sequence_length)

# questions = [[1,2,3,4,5], [1,2,3]]
# padded_questions, question_lengths = pad_sequences(questions, 0)
# print(padded_questions, question_lengths)