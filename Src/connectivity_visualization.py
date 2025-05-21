def read_lexicon(path):
    inputs, targets = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                devanagari, latin = parts[0], parts[1]
                inputs.append(latin)
                targets.append(devanagari)
    return inputs, targets

train_inp, train_out = read_lexicon('/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv')
val_inp,   val_out   = read_lexicon('/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv')

from collections import Counter
input_chars = sorted(list({c for s in train_inp for c in s}))
target_chars= sorted(list({c for s in train_out for c in s}))
input_vocab = {'<PAD>':0, '<SOS>':1, '<EOS>':2, **{c:i+3 for i,c in enumerate(input_chars)}}
target_vocab= {'<PAD>':0, '<SOS>':1, '<EOS>':2, **{c:i+3 for i,c in enumerate(target_chars)}}


import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, rnn_type='gru', dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers, nonlinearity='tanh',
                              batch_first=True, dropout=dropout)
    
    def forward(self, src, src_lengths=None):
        # src: (batch, seq_len) tensor of indices
        embedded = self.embedding(src)  # (batch, seq_len, emb_dim)
        # Optionally pack/unpack if using src_lengths
        outputs, hidden = self.rnn(embedded)  # outputs: (batch, seq_len, hidden_dim)
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, keys):
        # query: (batch, 1, hidden_dim), keys: (batch, seq_len, hidden_dim)
        # Compute scores:
        # score = v^T * tanh(Wa * query + Ua * keys)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = torch.softmax(scores.squeeze(-1), dim=-1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), keys)      # (batch, 1, hidden_dim)
        return context, weights  # context to use, and attention weights:contentReference[oaicite:12]{index=12}


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, rnn_type='gru', dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attn = BahdanauAttention(hidden_dim)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward_step(self, input_char, hidden, encoder_outputs):
        # input_char: (batch,1), hidden: last decoder hidden, encoder_outputs: (batch, seq_len, hidden)
        embedded = self.embedding(input_char)  # (batch,1,emb_dim)
        # Use last hidden state as query for attention:
        if isinstance(hidden, tuple):  # LSTM
            query = hidden[0][-1].unsqueeze(1)  # use last layer's hidden state
        else:
            query = hidden[-1].unsqueeze(1)
        context, attn_weights = self.attn(query, encoder_outputs)
        # Concatenate embedding and context, feed to RNN
        rnn_input = torch.cat([embedded, context], dim=2)  # (batch,1,emb+hid)
        output, new_hidden = self.rnn(rnn_input, hidden)
        output_char = self.out(output.squeeze(1))  # (batch, output_dim) logits
        return output_char, new_hidden, attn_weights  # return logits, new hidden, weights

    def forward(self, encoder_outputs, encoder_hidden, target_seq=None, max_len=20, teacher_forcing_ratio=0.5):
        # encoder_outputs: (batch, seq_len, hidden); encoder_hidden: initial hidden for decoder
        batch_size = encoder_outputs.size(0)
        outputs = []
        attn_weights = []
        # Initialize decoder input as <SOS> for each example
        decoder_input = torch.LongTensor([[target_vocab['<SOS>']]] * batch_size).to(encoder_outputs.device)
        hidden = encoder_hidden
        for t in range(max_len):
            output, hidden, weights = self.forward_step(decoder_input, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            attn_weights.append(weights.unsqueeze(1))
            # Decide next input: teacher forcing or max prob
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t].unsqueeze(1)  # Teacher forcing
            else:
                top1 = output.argmax(1).unsqueeze(1)
                decoder_input = top1.detach()
        outputs = torch.cat(outputs, dim=1)  # (batch, max_len, output_dim)
        attn_weights = torch.cat(attn_weights, dim=1)  # (batch, max_len, seq_len)
        return outputs, attn_weights



import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Seq2SeqDataset(Dataset):
    def __init__(self, inputs, targets, 
                 input_vocab, target_vocab, device='cpu'):
        """
        inputs: list of strings (Latin)
        targets: list of strings (Devanagari)
        """
        self.input_vocab  = input_vocab
        self.target_vocab = target_vocab
        self.device       = device

        # Pre-convert each string to a list of token indices (+ EOS for target)
        self.inputs_idx = [
            torch.tensor([input_vocab[c] for c in s], dtype=torch.long)
            for s in inputs
        ]
        self.targets_idx = [
            torch.tensor(
                [target_vocab['<SOS>']] +
                [target_vocab[c] for c in t] +
                [target_vocab['<EOS>']],
                dtype=torch.long
            )
            for t in targets
        ]

    def __len__(self):
        return len(self.inputs_idx)

    def __getitem__(self, idx):
        return self.inputs_idx[idx], self.targets_idx[idx]

def collate_fn(batch):
    """
    Pads input and target sequences to the length of the longest in the batch.
    Returns:
      - padded_inputs : LongTensor (batch, max_in_len)
      - padded_targets: LongTensor (batch, max_tgt_len)
    """
    inputs, targets = zip(*batch)
    # Pad with vocab['<PAD>']
    pad_in  = input_vocab['<PAD>']
    pad_tgt = target_vocab['<PAD>']

    padded_inputs  = pad_sequence(inputs,  batch_first=True, padding_value=pad_in)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_tgt)
    return padded_inputs.to(device), padded_targets.to(device)

# --- instantiate dataset & loader ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = Seq2SeqDataset(
    train_inp, train_out,
    input_vocab, target_vocab,
    device=device
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)


import torch
import torch.nn as nn
import torch.optim as optim
import random
num_epochs = 10
# (re)define your models, possibly with dropout=0.0 if num_layers=1
#encoder = EncoderRNN(len(input_vocab), emb_dim=256, hidden_dim=512,
                   #  num_layers=1, rnn_type='gru', dropout=0.0).to(device)
#decoder = AttnDecoderRNN(len(target_vocab), emb_dim=256, hidden_dim=512,
                       #  num_layers=1, rnn_type='gru', dropout=0.0).to(device)
encoder = EncoderRNN(len(input_vocab), emb_dim=256, hidden_dim=256,
                     num_layers=1, rnn_type='lstm', dropout=0.25).to(device)
decoder = AttnDecoderRNN(len(target_vocab), emb_dim=256, hidden_dim=256,
                         num_layers=1, rnn_type='lstm', dropout=0.25).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<PAD>'])

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        enc_out, enc_hidden = encoder(batch_inputs)
        # decoder input = all but last token; decoder target = all but first
        dec_in  = batch_targets[:, :-1]
        dec_out, _ = decoder(enc_out, enc_hidden,
                             target_seq=dec_in,
                             max_len=dec_in.size(1),
                             teacher_forcing_ratio=0.5)
        # compute loss against batch_targets[:,1:]
        logits = dec_out.view(-1, len(target_vocab))
        gold   = batch_targets[:,1:].contiguous().view(-1)
        loss   = criterion(logits, gold)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")


def translate_input(latin_str, encoder, decoder):
    # Prepare single example
    seq = torch.tensor([[input_vocab.get(c, input_vocab['<PAD>']) for c in latin_str]],
                       dtype=torch.long).to(device)
    enc_out, enc_hidden = encoder(seq)
    decoder_input = torch.tensor([[target_vocab['<SOS>']]], device=device)
    hidden = enc_hidden
    outputs = []
    attn_ws = []
    for t in range(max_output_len):
        logits, hidden, attn = decoder.forward_step(decoder_input, hidden, enc_out)
        prob = torch.softmax(logits, dim=1)  # (1,vocab)
        topv, topi = prob.topk(5)  # get top 5 next chars
        outputs.append(topi.cpu().numpy())  # (1,5)
        attn_ws.append(attn.cpu().detach().numpy())  # (1, enc_len)
        decoder_input = topi[:,0].view(1,1)  # use best for continuing this path
    # For simplicity, we only trace the single best path here.
    return outputs, attn_ws


test_inp, test_out = read_lexicon('/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv')

# 2) Compute maximum output length (in characters) you want to visualize.
#    We take the max length of any Devanagari word and add 1 for EOS.
max_output_len = max(len(word) for word in test_out) + 1

# 3) Build an index→char list for your target vocab
#    (so that `target_chars[i]` gives you the Devanagari character for index i)
inv_target_vocab = {idx: char for char, idx in target_vocab.items()}

from ipywidgets import interact, IntSlider
from IPython.display import HTML, display
import numpy as np, matplotlib.cm as cm

# pre-define colormap
cmap = cm.get_cmap('viridis')
def get_top5_and_attn(latin_str):
    """Returns:
        preds: List[str]       top-5 decoded strings
        attn:  List[np.array]  attention weights: shape (dec_len, enc_len)
    """
    # run your translate_input to get top5 idxs and attn_ws
    topk_outputs, attn_ws = translate_input(latin_str, encoder, decoder)
    # convert to strings
    preds = []
    for beam in range(5):
        chars = []
        for step in topk_outputs:
            idx = int(step[0, beam])
            ch  = inv_target_vocab[idx]
            if ch=='<EOS>': break
            chars.append(ch)
        preds.append(''.join(chars))
    # stack attns into array: (dec_len, enc_len)
    attn_matrix = np.vstack([w[0] for w in attn_ws])
    return preds, attn_matrix

def color_for_weight(w):
    r,g,b,a = cmap(float(w))
    return f'background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f});'

def build_full_table(latin, ground_truth, preds, attn, char_idx):
    """
    latin        : the input Latin string
    ground_truth : the true Hindi word (for reference)
    preds        : list of top-5 predicted Devanagari strings
    attn         : np.array (dec_len, enc_len)
    char_idx     : which output position to inspect
    """
    # Wrap in a div to force Devanagari font
    html = f"""
    <div style="font-family:'Noto Sans Devanagari',sans-serif; font-size:16px;">
      <p><b>Latin input:</b> {latin}<br>
         <b>Ground truth:</b> {ground_truth}</p>
      <table style="border-collapse:collapse; width:100%;">
        <tr>
          <th style="border:1px solid #666;padding:6px;">Prediction (pos {char_idx})</th>
          <th style="border:1px solid #666;padding:6px;">Attention weights over Latin input</th>
        </tr>
    """

    # For each of the top‐5 predictions
    for pred in preds:
        # 1) Highlight the target char in the prediction
        pred_spans = []
        for i, ch in enumerate(pred):
            if i == char_idx:
                # yellow background for the focused output char
                style_out = "background-color:#FFD54F;padding:2px;border-radius:3px;"
            else:
                style_out = ""
            pred_spans.append(f"<span style='{style_out}'>{ch}</span>")
        colored_pred = "".join(pred_spans) or "<EOS>"

        # 2) Grab the attention row (or zeros if out of range)
        if char_idx < attn.shape[0]:
            row = attn[char_idx]
        else:
            row = np.zeros(len(latin), dtype=float)

        # 3) Build the Latin string with per‐char opacity
        latin_spans = []
        for w, ch in zip(row, latin):
            # Map weight w∈[0,1] to alpha between 0.2 and 1.0
            alpha = 0.2 + 0.8 * float(w)
            # Yellow fill with variable opacity
            style_in = f"background-color: rgba(255, 235, 59, {alpha:.2f}); padding:2px; border-radius:3px;"
            latin_spans.append(f"<span style='{style_in}'>{ch}</span>")
        colored_latin = "".join(latin_spans)

        # 4) Add a row to the table
        html += f"""
        <tr>
          <td style="border:1px solid #666;padding:6px;text-align:center;">{colored_pred}</td>
          <td style="border:1px solid #666;padding:6px;text-align:center;">{colored_latin}</td>
        </tr>
        """

    html += "</table></div>"
    return html




@interact(
    example_idx=IntSlider(min=0, max=len(test_inp)-1, step=1, description='Example'),
    char_idx=IntSlider(min=0, max=max_output_len-1, step=1, description='Char idx')
)
def show_all_attention(example_idx=0, char_idx=0):
    latin  = test_inp[example_idx]
    truth  = test_out[example_idx]
    preds, attn = get_top5_and_attn(latin)          # as defined before
    html = build_full_table(latin, truth, preds, attn, char_idx)
    display(HTML(html))
