def reverse_sentence(sentence, up_to_k_words=5):
    """ Reverses a sentence up to k words

    Args:
        sentence (str): The sentence to reverse
        up_to_k_words (int, optional): The number of words to reverse


    Returns:
        str: The reversed sentence (up to k words)
    """
    return " ".join(sentence.split(" ")[::-1][:up_to_k_words])