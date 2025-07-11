from collections import defaultdict
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def create_index(self, text):
        """
        This method create the index by applying methods like BPE, Wordpiece etc
        """
        pass
    
    @abstractmethod
    def get_index(self):
        """
        This method returns the text token to idx mapping
        """
        pass
    
    @abstractmethod
    def get_reverse_index(self):
        """
        This method returns the token idx to text mapping
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        Based on the toneization technique, this method uses the index to generate the index.
        """
        pass

    @abstractmethod
    def decode(self, tokens):
        """
        Based on the toneization technique, this method uses the index to generate the index.
        """
        pass

class BytePairEncoding(Tokenizer):
    def __init__(self, max_vocab_size):
        """
        Init method for Byte Pair Encoding
        :param vocab_size: maximum size of the vocabulary
        """
        self.max_vocab_size = max_vocab_size
        self.id2token_index = None
        self.token2id_index = None
        self.pad_token = None
        self.merge_rules = []

    def _preprocess_text(self, texts):
        tokenized_text = {}
        initial_vocab = set()
        for i, text in enumerate(texts):
            tokenized_text[i] = list(text.strip())
            initial_vocab.update(tokenized_text[i])
        return tokenized_text, initial_vocab
    
    def _get_most_frequent_pair(self, tokenized_text):
        token_freq_map = defaultdict(int)
        for i, tokens in tokenized_text.items():
            for token_1, token2 in zip(tokens, tokens[1:]):
                token_freq_map[(token_1, token2)] += 1
        if not token_freq_map:
            return None, None
        # max_frequency = max(token_freq_map.values())
        # max_freq_pair = [i for i in token_freq_map if token_freq_map[i]==max_frequency][0]
        max_freq_pair = max(token_freq_map, key=lambda x: token_freq_map[x])
        return max_freq_pair, token_freq_map[max_freq_pair]
    
    def _merge_tokens(self, tokenized_text, max_freq_pair):
        """
        This method merges the occurrence of max_freq_pair tokens in tokenized_text
        :param tokenized_text: - A dictionary of index to list of tokens mapping
        :param max_freq_pair: - a tuple of two tokens which have the max frequency of occurring
        """
        for i, tokens in tokenized_text.items():
            updated_tokens = []
            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == max_freq_pair[0] and tokens[j + 1] == max_freq_pair[1]:
                    updated_tokens.append(tokens[j] + tokens[j + 1])
                    j += 2
                else:
                    updated_tokens.append(tokens[j])
                    j += 1
            
            if j == len(tokens) - 1:  # Handle the last token if it doesn't form a pair with max_freq_pair
                updated_tokens.append(tokens[-1])
            
            tokenized_text[i] = updated_tokens
        return tokenized_text


    def create_index(self, texts):
        """
        Step:
        1. Start with an initial vocab
        2. Find the frequency of pair of tokens in the input texts based the vocab
        3. Add the most frequently occuring pair in the vocab
        4. Repeat step 2-3 until vocab_size is reached
        
        :param text: The text based on which we are generating the vocab
        :param initial_vocab: The initial vocabulary 

        """
        tokenized_text, vocab = self._preprocess_text(texts)
        self.merge_rules = []
        prev_vocab_size = len(vocab)
        while len(vocab)<self.max_vocab_size:
            max_freq_pair, max_frequency = self._get_most_frequent_pair(tokenized_text)
            if not max_freq_pair:
                break
            self.merge_rules.append(max_freq_pair)
            tokenized_text = self._merge_tokens(tokenized_text, max_freq_pair)
            vocab.add("".join(max_freq_pair))
        
        self.id2token_index = {i: token for i, token in enumerate(vocab)}
        self.token2id_index = {token: i for i, token in enumerate(vocab)}
        return self.token2id_index
    
    def get_index(self):
        return self.token2id_index

    def get_reverse_index(self):
        return self.id2token_index

    def encode(self, text):
        """
        Based on the toneization technique, this method uses the index to generate the index.
        """
        tokens = {0: list(text.strip())}
        for rule in self.merge_rules:
            tokens = self._merge_tokens(tokens, rule)
        return [self.token2id_index[token] for token in tokens[0]]
         
        
    def decode(self, tokens):
        return [self.id2token_index[token_idx] for token_idx in tokens]


bpe = BytePairEncoding(20)
corpus = [
    "low",
    "lower",
    "lowest",
    "newer",
    "newest",
    "wide",
    "wider",
    "widest",
    "high",
    "higher",
    "highest",
    "bright",
    "brighter",
    "brightest",
    "dark",
    "darker",
    "darkest",
    "slow",
    "slower"
]
index = bpe.create_index(corpus)
print("Encoding \"slowest\"", bpe.encode("slowest"))
print("Decoding [19, 2, 10, 18, 1]", bpe.decode([19, 2, 10, 18, 1]))