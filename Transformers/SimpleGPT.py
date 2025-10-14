import os
import torch
import torch.nn as nn
import numpy as np
import random
import multiprocessing
from torchviz import make_dot

def set_seed(seed_value=42):
    """Sets the seed for reproducibility across different libraries."""

    # PYTHONHASHSEED - It is an environment variable in Python that sets the seed for hash randomization. Hash randomization affects the ordering of elements in things like dictionaries, sets, and anything that relies on hash tables internally. PYTHONHASHSEED doesn't directly affect PyTorch computations like random tensors or model training. But it can indirectly affect training/evaluation in cases where you shuffle datasets that are organized as a dictionary or set or  split datasets based on a random ordering of keys.

    # Since Python 3.7 dict preserves the insertion order i.e. the order of keys you insert into a dict is the order you will see them when you iterate over the dict later. While dict iteration order is now stable (insertion-based), the hash values themselves (i.e., what hash(key) returns) are still randomized if PYTHONHASHSEED isn't set. If any of the pytorch internal sampling, tokenization or workers shuffling depend on this hash, it will result in non-deterministic behaviour. 
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multiple GPUs.
    

    # CuDNN is CUDA Deep Neural Network library made by NVIDIA. It’s a low-level GPU-accelerated library that PyTorch, TensorFlow, and other frameworks use under the hood to run things like convolutions, activation functions, RNNs). CuDNN has multiple implementations for convolutions e.g. GEMM-based, Winograd etc. Each of these are better for different scenarios such as some are faster for smaller kernels while others use less memory. Whenever CuDNN convolutions are called, a benchmarking is called to find the implementation that runs the fastest for the current set of parameters sizes. This implementation is then used in all the future calls. Whenever the parameters sizes changes the benchmark is rerun, thus introducing the non-determinism(difficult to compare the performance of different sizes). When disabled, the safest algorithm is chosen always which is deterministic across all runs but it comes at a cost of maybe diminished speed. Hence, if input sizes are fixed, use benchmarking to speed up but if input sizes keep changing disable benchmarking
    torch.backends.cudnn.benchmark = False

    # The various CuDNN implementations themselves might be using RNGs(Random Number Generators). Adding torch.backends.cudnn.deterministic = True will ensure the algorithm behaves same every time. 
    torch.backends.cudnn.deterministic = True
    
    print(f"Random seed set as {seed_value}")

# Usage
set_seed(42)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_weights = nn.Embedding(vocab_size, d_model, dtype=torch.float32)
    
    def forward(self, x):
        """
        This method takes in a batch of input tokens of size B X N and returns embeddings corresponding to them of size B X N X d_model
        :param x: Input tokens of shape B X N 
        """
        return self.embedding_weights(x)
        
# Ideally this is not necessary but we will keep it in case we want to plug in another type of initialization

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_embeddings = nn.Embedding(self.seq_len, self.d_model, dtype=torch.float32)
    
    def forward(self, x):
        return self.pos_embeddings(x)
    
class TransformerLayer(nn.Module):
    """
    This class performs the following equations
    x = x + Dropout(MHA(Layer_Norm1(x)))
    x = x + Dropout(FeedForwardNetwork(Layer_Norm2(x)))
    """
    def __init__(self, d_model, num_heads, dropout_p_mha, dropout_p_ffn):
        super().__init__()
        self.MHA = MultiHeadAttention(num_heads, d_model)
        self.layer_norm_1 = LayerNorm(d_model)
        self.dropout_1 = Dropout(dropout_p_mha)
        self.ffn  = FeedForwardNetwork(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout_2 = Dropout(dropout_p_ffn)

    def forward(self, inputs):
        """
        Takes input of size B X N X d_model and returns output of size B X N X d_model
        """
        x, attention_mask = inputs[0], inputs[1]
        x = x + self.dropout_1(self.MHA(self.layer_norm_1(x), attention_mask))
        x = x + self.dropout_2(self.ffn(self.layer_norm_2(x)))
        return [x, attention_mask]

class LayerNorm(nn.Module):
    """
    This class performs the Layer Normalization. 
    x_norm = scale*(x-mean)/std_dev + shift where scale and shift are learable parameter of size (d_model)
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(self.d_model, dtype=torch.float32))
        self.shift = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        """
        Takes input of size B X N X d_model and performs normalization across the last dimension.
        :param x: Tensor of shape B X N X d_model
        """
        return (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + self.eps) * self.scale + self.shift



class FeedForwardNetwork(nn.Module):
    """
    This class applies a feed forward network over the input. It essentially passes the input through two linear layers
        1. First Layer d_model X 4*d_model
        2. Non-linearity - Relu, Gelu etc
        4. Second Layer - 4*d_model X d_model
    Takes input of size B X N X d_model and applies a feed forward network over it and returns an output of size B X N X d_model

    Few models started using 8/3*d_model as hidden_dim
    """
    def __init__(self, d_model, hidden_dim=None):
        """
        """
        super().__init__()
        hidden_dim = hidden_dim or 4*d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(4*d_model, d_model, dtype=torch.float32)
        )
    
    def forward(self, x):
        """
        Input and Output are of dimension B X N X d_model 
        """
        return self.ffn(x)


class Dropout(nn.Module):
    """
    """
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        """
        Takes input of size B X N X d_model and applies a Dropout layer over it and returns an output of size B X N X d_model
        """
        bernoulli_distribution = torch.distributions.Bernoulli(probs = 1-self.prob)
        mask = ~bernoulli_distribution.sample(x.shape).bool()
        # alternative
        # mask = (torch.rand_like(x) < self.prob)  
        x.masked_fill_(mask, 0)
        x = x/(1-self.prob) # inverted dropout
        return x

def stable_softmax(x, dim=-1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    return torch.softmax(x, dim=dim)

class MultiHeadAttention(nn.Module):
    """
    This class applies the MultiHeadAttention Layer over the input. The Self-Attention mechanism is run in parallel each time with a different set of learned parameters for query, key and value projections. If the number of heads are represented num_heads, to keep the input and output dimensions constant we compute the the hidden size d_k = d_model/num_heads. All the outputs from the head are concatenated and passed through a Linear Layer and a dropout is applied on them.
    MHA(x) = Linear(Concat([Self_Attention(x)_1, ...., Self_Attention(x)_num_heads]))
    """
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if self.d_model%num_heads:
            raise Exception("d_model {d_model} is not divisible by num_heads {num_heads}")
        self.d_k = self.d_model/num_heads
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)
        self.out = nn.Linear(self.d_model, self.d_model, dtype=torch.float32)


    def forward(self, x, attention_mask):
        """
        Takes input of size B X N X d_model and applies a MHA layer over it and returns an output of size B X N X d_model. Uses attention_mask to stop computing attention for padding tokens. Attention mask is of shape B X N
        Steps:
        1. Compute Query Key and Value Vectors
        2. Split the Q, K, V tensors to inclue num_heads dimension B X N X d_model -> B X num_heads X N X d_k
        3. Compute Attention scores
        4. Create a mask combining both causal masks and attention mask
        4. Compute attention weights using softmax
        5. Multiply with values and reshape
        6. Apply final linear layer and return
        """
        batch_size, seq_length, _ = x.shape
        queries, keys, values = self.W_Q(x), self.W_K(x), self.W_V(x)
        queries = queries.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2) # B X num_heads X N X d_k
        keys = keys.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2) # B X num_heads X N X d_k
        values = values.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2) # B X num_heads X N X d_k

        # B X num_heads X N X d_k @ B X num_heads X d_k X N
        attention_scores = queries @ keys.transpose(2, 3) # B X num_heads X N X N

        causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=attention_scores.dtype), diagonal=1).bool()
        attention_scores.masked_fill_(causal_mask, -torch.inf)

        attention_mask = (1-attention_mask).bool().unsqueeze(1).unsqueeze(2) # B X 1 X 1 X N
        attention_scores.masked_fill_(attention_mask, -torch.inf)
        
        attention_weights = stable_softmax(attention_scores/(self.d_k**0.5)) # B X num_heads X N X N
        contexts = attention_weights @ values # B X num_heads X N X d_k
        contexts = contexts.transpose(1, 2).reshape(batch_size, seq_length, -1) # B X N X d_model
        contexts = self.out(contexts) # B X N X d_model
        return contexts


class SimpleGPT(nn.Module):
    """
    x = x + Dropout(MHA(Layer_Norm1(x)))
    x = x + Dropout(FeedForwardNetwork(Layer_Norm2(x)))
    """
    def __init__(self, vocab_size, max_seq_length, num_layers, d_model, num_heads, dropout_p_ffn, dropout_p_mha):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(seq_len=max_seq_length, d_model=d_model)
        self.transformer_blocks = nn.Sequential(*[TransformerLayer(d_model, num_heads, dropout_p_ffn, dropout_p_mha) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)
        self.out_head = nn.Parameter(self.embedding.embedding_weights.weight)

    def forward(self, input_ids, position_ids, attention_mask):
        """
        Takes input of size B X N text token indices and attention maps and applies the Transformer over it and returns an output of size B X N X V
        """
        x = self.embedding(input_ids) + self.positional_encoding(position_ids) # B X N X d_model
        x, _ = self.transformer_blocks([x, attention_mask]) # B X N X d_model
        x = self.layer_norm(x) # B X N X d_model
        logits = x @ self.out_head.T # B X N X V
        return logits
    
class TransformerLayerModified(nn.Module):
    """
    This class performs the following equations
    x = x + Dropout(MHA(Layer_Norm1(x)))
    x = x + Dropout(FeedForwardNetwork(Layer_Norm2(x)))
    """
    def __init__(self, d_model, num_heads, dropout_p_mha, dropout_p_ffn):
        super().__init__()
        self.MHA = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_p_mha)
        self.ffn  = FeedForwardNetwork(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout_p_ffn)

    def forward(self, inputs):
        """
        Takes input of size B X N X d_model and returns output of size B X N X d_model
        """
        x, attention_mask = inputs[0], inputs[1]
        x = x + self.dropout_1(self.MHA(self.layer_norm_1(x), attention_mask))
        x = x + self.dropout_2(self.ffn(self.layer_norm_2(x)))
        return [x, attention_mask]


