from SimpleGPT import *

vocab_size = 50
max_seq_length = 8 
num_layers = 2 
d_model = 4 
num_heads = 2 
dropout_p_ffn = dropout_p_mha = 0.4

inp_emb = InputEmbedding(vocab_size, d_model)
pos_enc = PositionalEncoding(max_seq_length, d_model)
trfm_layer = TransformerLayer(d_model, num_heads, dropout_p_mha, dropout_p_ffn)
layer_norm = LayerNorm(d_model)
ffn = FeedForwardNetwork(d_model)
dropout = Dropout(dropout_p_ffn)
mha = MultiHeadAttention(num_heads, d_model)
gpt = SimpleGPT(vocab_size, max_seq_length, num_layers, d_model, num_heads, dropout_p_ffn, dropout_p_mha)

batch_size = 2
seq_len = 8

# max seq len, full attention mask
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
position_ids = torch.arange(seq_len, dtype=torch.long).repeat(batch_size, 1)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long) 

emb = inp_emb(input_ids)
print(emb.shape)
pos_emb = pos_enc(position_ids)
print(pos_emb.shape)

print(emb[0][0][:])
# Python standard dev computed sample variance i.e. /(n-1)
emb_norm = layer_norm(emb)
print("After normalization")
print(emb_norm[0][0][:])

ffn_out = ffn(emb)
print(ffn_out.shape)

dropout_out = dropout(emb)
print(dropout_out.shape)


logits = mha(emb+pos_emb, attention_mask)
print(logits.shape)

layer_output = trfm_layer([emb, attention_mask])
print(layer_output[0].shape)


final_output = gpt(input_ids, position_ids, attention_mask)
print(final_output.shape)

# Total number of parameters
# from torchsummary import summary
# summary(transformer, ( 1, 1024, 768))
vocab_size = 25000
max_seq_length = 1024
d_model = 768
num_layers = 8
num_heads = 12
gpt = SimpleGPT(vocab_size, max_seq_length, num_layers, d_model, num_heads, dropout_p_ffn, dropout_p_mha)
# transformer.named_parameters()
# total_params = 0
def count_parameters(model):
    total_params = sum(p.numel() for name, p in model.named_parameters() if "embedding.embedding_weights.weight" not in name)
    return total_params
tot_params = count_parameters(gpt)
print(f"Num params: {tot_params/10**6}M")



layer_output = trfm_layer([emb, attention_mask])
print(layer_output[0].shape)
make_dot(layer_output[0].mean(), params=dict(gpt.named_parameters()))


##############################.    TRAINING LOOP. #############################################
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
position_ids = torch.arange(seq_len, dtype=torch.long).repeat(batch_size, 1)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long) 
targets = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.long)
# Trying to overfit the transformer to check if it has learning abilities
num_epochs = 2000
gpt = SimpleGPT(vocab_size, max_seq_length, num_layers, d_model, num_heads, dropout_p_ffn, dropout_p_mha)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(gpt.parameters(), lr = 0.1)
gpt.train()
training_losses = []
for i in range(num_epochs):
    logits = gpt(input_ids, position_ids, attention_mask)
    loss = criterion(logits[:, -1, :], targets.view(-1))
    training_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






