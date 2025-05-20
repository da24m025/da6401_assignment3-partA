import os
import random
import numpy as np
import tensorflow as tf
import wandb

# ─── FORCE EAGER EXECUTION ────────────────────────────────────────────────
tf.config.run_functions_eagerly(True)

# ─── Reproducibility & API Key ───────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['WANDB_API_KEY'] = "d3cfbe96f8cae1276193dd36e66c72f41b1ac020"
wandb.login()  # Ensure you're authenticated before using wandb

# ─── Imports ──────────────────────────────────────────────────────────────
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, GRU, Dense,
    AdditiveAttention, TimeDistributed, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

# ─── Paths & Special Tokens ───────────────────────────────────────────────
START, END, PAD = '\t', '\n', ' '
DATA_DIR = '/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons'
TRAIN = os.path.join(DATA_DIR, 'hi.translit.sampled.train.tsv')
DEV   = os.path.join(DATA_DIR, 'hi.translit.sampled.dev.tsv')
TEST  = os.path.join(DATA_DIR, 'hi.translit.sampled.test.tsv')

# ─── Data Loading & Vectorization ────────────────────────────────────────
def read_tsv(path):
    with open(path, encoding='utf8') as f:
        lines = [l.strip().split('\t') for l in f if l.strip()]
    srcs = [x[1] for x in lines]
    tgts = [x[0] for x in lines]
    return srcs, tgts

train_src, train_tgt = read_tsv(TRAIN)
dev_src,   dev_tgt   = read_tsv(DEV)
test_src,  test_tgt  = read_tsv(TEST)

# build source vocab
char2idx_src = {PAD:0}
for seq in train_src + dev_src:
    for ch in seq:
        char2idx_src.setdefault(ch, len(char2idx_src))
idx2char_src = {i:c for c,i in char2idx_src.items()}
max_len_src = max(len(s) for s in train_src + dev_src)

# build target vocab
char2idx_tgt = {START:0, END:1, PAD:2}
for seq in train_tgt + dev_tgt:
    for ch in seq:
        char2idx_tgt.setdefault(ch, len(char2idx_tgt))
idx2char_tgt = {i:c for c,i in char2idx_tgt.items()}
max_len_tgt = max(len(s) for s in train_tgt + dev_tgt) + 2

def vectorize(src, tgt=None):
    X = np.zeros((len(src), max_len_src), dtype='int32')
    for i, s in enumerate(src):
        for t,ch in enumerate(s):
            X[i,t] = char2idx_src[ch]
    if tgt is None:
        return X
    dec_in = np.zeros((len(tgt), max_len_tgt), dtype='int32')
    dec_out = np.zeros((len(tgt), max_len_tgt, len(char2idx_tgt)), dtype='float32')
    for i, s in enumerate(tgt):
        dec_in[i,0] = char2idx_tgt[START]
        for t,ch in enumerate(s):
            dec_in[i,t+1] = char2idx_tgt[ch]
        dec_in[i,len(s)+1] = char2idx_tgt[END]
        for t in range(1, max_len_tgt):
            dec_out[i, t-1, dec_in[i,t]] = 1.0
    return X, dec_in, dec_out

X_train, y_in_train, y_out_train = vectorize(train_src, train_tgt)
X_dev,   y_in_dev,   y_out_dev   = vectorize(dev_src,   dev_tgt)
X_test,  y_in_test,  y_out_test  = vectorize(test_src,  test_tgt)

# ─── Seq2Seq + Attention Model ───────────────────────────────────────────
def build_model(enc_vocab, dec_vocab,
                emb_dim=64, hid=64,
                cell_type="LSTM", dropout=0.0):
    enc_inputs = Input(shape=(None,), name="enc_in")
    enc_emb    = Embedding(enc_vocab, emb_dim, mask_zero=False, name="enc_emb")(enc_inputs)
    if cell_type == "LSTM":
        enc_out, h, c = LSTM(hid, return_sequences=True, return_state=True,
                             dropout=dropout, name="encoder")(enc_emb)
        enc_states = [h, c]
    else:
        enc_out, h = GRU(hid, return_sequences=True, return_state=True,
                         dropout=dropout, name="encoder")(enc_emb)
        enc_states = h  # not a list

    dec_inputs = Input(shape=(None,), name="dec_in")
    dec_emb    = Embedding(dec_vocab, emb_dim, mask_zero=False, name="dec_emb")(dec_inputs)
    if cell_type == "LSTM":
        dec_out, _, _ = LSTM(hid, return_sequences=True, return_state=True,
                             dropout=dropout, name="decoder")(dec_emb, initial_state=enc_states)
    else:
        dec_out, _ = GRU(hid, return_sequences=True, return_state=True,
                         dropout=dropout, name="decoder")(dec_emb, initial_state=[enc_states])

    attn_layer = AdditiveAttention(name="attention")
    context, scores = attn_layer([dec_out, enc_out], return_attention_scores=True)
    concat = Concatenate(name="concat")([context, dec_out])
    outputs = TimeDistributed(Dense(dec_vocab, activation="softmax"), name="output_dense")(concat)

    model = Model([enc_inputs, dec_inputs], outputs, name="seq2seq_attn")
    return model

# ─── Train / Sweep ─────────────────────────────────────────────────────────
def train():
    run = wandb.init()
    cfg = run.config

    model = build_model(
        enc_vocab=len(char2idx_src),
        dec_vocab=len(char2idx_tgt),
        emb_dim=cfg.emb_dim,
        hid=cfg.hid_size,
        cell_type=cfg.cell_type,
        dropout=cfg.dropout
    )
    model.compile(
        optimizer=Adam(cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    log_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: wandb.log({
        "train_loss":  logs["loss"],
        "train_acc":   logs["accuracy"],
        "val_loss":    logs["val_loss"],
        "val_acc":     logs["val_accuracy"]
    }, step=epoch))

    model.fit(
        [X_train, y_in_train], y_out_train,
        validation_data=([X_dev, y_in_dev], y_out_dev),
        batch_size=cfg.batch_size,
        epochs=cfg.get("epochs", 20),
        callbacks=[EarlyStopping("val_loss", patience=3, restore_best_weights=True), log_cb],
        verbose=2
    )

    preds = model.predict([X_test, y_in_test])
    os.makedirs("predictions_attention", exist_ok=True)
    np.savez("predictions_attention/preds.npz", preds=preds)

    test_acc = sum(
        1 for i,tgt in enumerate(test_tgt)
        if "".join(
            idx2char_tgt[idx] for idx in np.argmax(preds[i],axis=-1)
            if idx2char_tgt[idx]!=END
        ) == tgt
    ) / len(test_tgt)
    wandb.log({"test_accuracy": test_acc})
    run.finish()

sweep_config = {
    "method":"bayes",
    "metric":{"name":"val_loss","goal":"minimize"},
    "parameters":{
        "emb_dim":      {"values":[32,64,128]},
        "hid_size":     {"values":[32,64,128]},
        "cell_type":    {"values":["LSTM","GRU"]},
        "dropout":      {"values":[0.0,0.2]},
        "learning_rate":{"distribution":"log_uniform_values","min":1e-4,"max":1e-2},
        "batch_size":   {"values":[32,64]},
        "epochs":       {"value":8}
    }
}

if __name__=="__main__":
    sweep_id = wandb.sweep(sweep_config, project="attn_seq2seq")
    wandb.agent(sweep_id, function=train, count=20) 
