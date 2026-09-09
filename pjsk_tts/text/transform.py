"""from https://github.com/keithito/tacotron"""


def cleaned_text_to_sequence(cleaned_text, symbols):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence
