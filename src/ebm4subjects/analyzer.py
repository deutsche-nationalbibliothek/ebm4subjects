import re

import nltk.data


class EbmAnalyzer:
    def __init__(self, tokenizer_name: str) -> None:
        try:
            nltk.data.find(tokenizer_name)
        except LookupError as error:
            if tokenizer_name in str(error):
                nltk.download(tokenizer_name)
            else:
                raise

        self.tokenizer = nltk.data.load(tokenizer_name)

    def tokenize_sentences(self, text: str) -> list[str]:
        text = re.sub("\.{4,}", ". ", text)
        return self.tokenizer.tokenize(text)
