import os, random, numpy as np, tensorflow as tf, wandb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, GRU, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from wandb.integration.keras import WandbCallback

# ─── Reproducibility & API key ────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["WANDB_API_KEY"] = "1a0371e8845a9011d0637dc53664d347c52f64b6"

# ─── Special tokens & data paths ───────────────────────────────────────────────
START_CHAR, END_CHAR, BLANK_CHAR = "\t", "\n", " "
DATA_DIR    = "/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons"
TRAIN, VAL  = os.path.join(DATA_DIR, "hi.translit.sampled.train.tsv"), os.path.join(DATA_DIR, "hi.translit.sampled.dev.tsv")

# ─── Read & build vocab ────────────────────────────────────────────────────────
def read_data(path):
    lines = [l.split("\t") for l in open(path,encoding="utf-8") if l]
    return [x[1] for x in lines], [x[0] for x in lines]

train_i, train_o = read_data(TRAIN)
val_i,   val_o   = read_data(VAL)

# encoder vocab
in_enc, in_dec = {}, []
for s in train_i + val_i:
    for ch in s:
        if ch not in in_enc:
            in_enc[ch] = len(in_dec); in_dec.append(ch)
in_enc.setdefault(BLANK_CHAR, len(in_dec)); in_dec.append(BLANK_CHAR)
max_enc = max(len(s) for s in train_i+val_i)

# decoder vocab
tgt_enc, tgt_dec = {START_CHAR:0}, [START_CHAR]
for t in train_o + val_o:
    for ch in t:
        if ch not in tgt_enc:
            tgt_enc[ch] = len(tgt_dec); tgt_dec.append(ch)
tgt_enc[END_CHAR] = len(tgt_dec); tgt_dec.append(END_CHAR)
tgt_enc.setdefault(BLANK_CHAR, len(tgt_dec)); tgt_dec.append(BLANK_CHAR)
max_dec = max(len(t) for t in train_o+val_o) + 2

# ─── Data→array ───────────────────────────────────────────────────────────────
def proc(inp, tim_enc, enc_map, tgt=None, tim_dec=None, dec_map=None):
    X = np.array([[enc_map[ch] for ch in s] + [enc_map[BLANK_CHAR]]*(tim_enc-len(s)) for s in inp],dtype="int32")
    if tgt is None: return X, None, None
    N, V = len(tgt), len(dec_map)
    dec_in = np.zeros((N,tim_dec), dtype="int32")
    dec_out = np.zeros((N,tim_dec,V),dtype="float32")
    for i, s in enumerate(tgt):
        dec_in[i,0] = dec_map[START_CHAR]
        for t,ch in enumerate(s):
            dec_in[i,t+1] = dec_map[ch]
        dec_in[i,len(s)+1] = dec_map[END_CHAR]
        for t in range(1,tim_dec):
            dec_out[i,t-1,dec_in[i,t]] = 1.0
    return X, dec_in, dec_out

train_enc, train_dec_in, train_dec_out = proc(train_i, max_enc, in_enc, train_o, max_dec, tgt_enc)
val_enc,   val_dec_in,   val_dec_out   = proc(val_i,   max_enc, in_enc, val_o,   max_dec, tgt_enc)

# ─── Model builder ─────────────────────────────────────────────────────────────
def create_model(enc_vocab, dec_vocab,
                 emb_dim=64, enc_layers=1, dec_layers=1,
                 hid_size=64, cell_type="LSTM",
                 dropout=0.0, rec_dropout=0.0):
    # FORCE rec_dropout=0 for RNN/GRU
    if cell_type != "LSTM":
        rec_dropout = 0.0
        dropout = 0.0

    cell_map = {"RNN":SimpleRNN, "GRU":GRU, "LSTM":LSTM}
    RNNCell  = cell_map[cell_type]

    enc_in  = Input((None,), name="encoder_in")
    emb_e   = Embedding(enc_vocab, emb_dim, name="emb_e")(enc_in)
    x, *enc_states = RNNCell(hid_size, return_sequences=True, return_state=True,
                              dropout=dropout, recurrent_dropout=rec_dropout,
                              name="enc_0")(emb_e)
    for i in range(1,enc_layers):
        x, *enc_states = RNNCell(hid_size, return_sequences=True, return_state=True,
                                  dropout=dropout, recurrent_dropout=rec_dropout,
                                  name=f"enc_{i}")(x)

    dec_in  = Input((None,), name="decoder_in")
    emb_d   = Embedding(dec_vocab, emb_dim, name="emb_d")(dec_in)
    y, *dec_states = RNNCell(hid_size, return_sequences=True, return_state=True,
                              dropout=dropout, recurrent_dropout=rec_dropout,
                              name="dec_0")(emb_d, initial_state=enc_states)
    for i in range(1,dec_layers):
        y, *dec_states = RNNCell(hid_size, return_sequences=True, return_state=True,
                                  dropout=dropout, recurrent_dropout=rec_dropout,
                                  name=f"dec_{i}")(y, initial_state=dec_states)

    out = Dense(dec_vocab, activation="softmax", name="out")(y)
    return Model([enc_in, dec_in], out, name="seq2seq")

# ─── Exact‐match callback ──────────────────────────────────────────────────────
class SeqAcc(Callback):
    def __init__(self, X, D_in, truths, dec_map):
        self.X, self.D_in, self.truths, self.dec_map = X, D_in, truths, dec_map

    def on_epoch_end(self, epoch, logs=None):
        P = self.model.predict([self.X,self.D_in], verbose=0)
        matches=0
        for i, t in enumerate(self.truths):
            idxs = P[i].argmax(-1)
            out = []
            for ix in idxs:
                ch = self.dec_map[ix]
                if ch==END_CHAR: break
                out.append(ch)
            if "".join(out)==t: matches+=1
        acc=matches/len(self.truths)
        print(f" — seq_val_accuracy: {acc:.4f}")
        wandb.log({"seq_val_accuracy":acc}, commit=False)

# ─── Log train/val loss & acc ─────────────────────────────────────────────────
class LossAccLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs contains: loss, accuracy, val_loss, val_accuracy, plus seq_val_accuracy
        l = logs.get("loss")
        a = logs.get("accuracy")
        vl= logs.get("val_loss")
        va= logs.get("val_accuracy")
        print(f" → loss:{l:.4f}, acc:{a:.4f}, val_loss:{vl:.4f}, val_acc:{va:.4f}")
        wandb.log({
            "loss":l, "accuracy":a,
            "val_loss":vl, "val_accuracy":va
        }, commit=False)

# ─── Sweep config & runner ────────────────────────────────────────────────────
sweep_cfg = {
  "name":"seq2seq_beam_sweep","method":"bayes",
  "metric":{"name":"seq_val_accuracy","goal":"maximize"},
  "parameters":{
    "emb_dim":     {"values":[16,32,64,128]},
    "enc_layers":  {"values":[1,2]},
    "dec_layers":  {"values":[1]}, 
    "hid_size":    {"values":[32,64,128]},
    "cell_type":   {"values":["RNN","GRU","LSTM"]},
    "dropout":     {"values":[0.0,0.2, 0.4]},
    "rec_dropout": {"values":[0.0,0.2, 0.4]},  # only used by LSTM
    "learning_rate":{"distribution":"log_uniform_values","min":1e-4,"max":1e-2},
    "batch_size":  {"values":[32,64]}
  }
}

def train_with_sweep():
    run = wandb.init(project="assignment3", reinit=True)
    cfg = run.config

    model = create_model(
      enc_vocab=len(in_dec),
      dec_vocab=len(tgt_dec),
      emb_dim=cfg.emb_dim,
      enc_layers=cfg.enc_layers,
      dec_layers=cfg.dec_layers,
      hid_size=cfg.hid_size,
      cell_type=cfg.cell_type,
      dropout=cfg.dropout,
      rec_dropout=cfg.rec_dropout
    )

    # ← add accuracy metric here
    model.compile(
      optimizer=Adam(cfg.learning_rate),
      loss="categorical_crossentropy",
      metrics=["accuracy"]
    )

    callbacks = [
      EarlyStopping("val_loss",patience=3,restore_best_weights=True),
      LossAccLogger(),                 # log/print loss & acc
      SeqAcc(val_enc, val_dec_in, val_o, tgt_dec),
      WandbCallback(monitor="seq_val_accuracy", save_model=False, save_graph=False)
    ]

    model.fit(
      [train_enc, train_dec_in], train_dec_out,
      validation_data=([val_enc,val_dec_in], val_dec_out),
      batch_size=cfg.batch_size,
      epochs=10,
      callbacks=callbacks,
      verbose=2
    )

    run.finish()

if __name__=="__main__":
    sweep_id = wandb.sweep(sweep_cfg, project="assignment3")
    print("Sweep registered:", sweep_id)
    wandb.agent(sweep_id, function=train_with_sweep, count=25)

