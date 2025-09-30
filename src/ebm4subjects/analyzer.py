import re

import nltk.data


class EbmAnalyzer:
    """
    A class for tokenizing text using NLTK.

    Attributes:
        tokenizer (nltk.tokenize.TokenizerI): The loaded NLTK tokenizer.

    Methods:
        - tokenize_sentences: Tokenizes the input text into sentences

    Raises:
        LookupError: If the specified tokenizer is not found.
    """

    def __init__(self, tokenizer_name: str) -> None:
        """
        Initializes the EbmAnalyzer with the specified tokenizer.

        Args:
            tokenizer_name (str): The name of the NLTK tokenizer to use.

        Raises:
            LookupError: If the specified tokenizer is not found.
        """
        # Attempt to find the tokenizer
        try:
            nltk.data.find(tokenizer_name)
        # If the tokenizer is not found, try to download it
        except LookupError as error:
            if tokenizer_name in str(error):
                nltk.download(tokenizer_name)
            else:
                raise

        # Load the tokenizer
        self.tokenizer = nltk.data.load(tokenizer_name)

    def tokenize_sentences(self, text: str) -> list[str]:
        """
        Tokenizes the input text into sentences using the loaded tokenizer.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str]: A list of tokenized sentences.
        """
        # Replace multiple periods by a singel one
        # Necessary to work properly with some tables of contents
        text = re.sub(r"\.{4,}", ". ", str(text))
        # Tokenize the text and return it
        return self.tokenizer.tokenize(text)
