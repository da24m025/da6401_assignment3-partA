#!/usr/bin/env python3
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, SimpleRNN, GRU, LSTM,
    Dense, AdditiveAttention, Concatenate, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback

# ─── Attention heatmap utility ───────────────────────────────────────────────
def plot_heatmaps(attn_model, num_examples=9):
    os.makedirs("heatmaps", exist_ok=True)
    # select first N examples
    for i in range(num_examples):
        src = test_inp[i]
        tgt = test_tgt[i]
        x_enc = test_enc[i:i+1]
        x_dec = np.zeros((1, max_dec), dtype="int32")
        x_dec[0,0] = dec_map[START_CHAR]

        # run through encoder & decoder with attention scores
        # we need a model that outputs the attention scores from the `attention` layer
        # keras functional: find that layer by name
        attn_layer = attn_model.get_layer("attention")
        # build sub‑model: inputs = full model inputs, outputs = attention scores
        sub = Model(attn_model.inputs,
                    attn_layer([attn_model.get_layer("dec_in").output,
                                attn_model.get_layer("enc_in").output],
                               return_attention_scores=True)[1])

        # now get scores: shape (1, T_dec, T_enc)
        scores = sub.predict([x_enc, x_dec], verbose=0)[0]  # drop batch

        plt.figure(figsize=(6,4))
        plt.imshow(scores[:len(src)+1, :len(src)], aspect="auto", cmap="viridis")
        plt.colorbar(); plt.title(f"Input→Output Alignment\n“{src}”→“{tgt}”")
        plt.xlabel("Encoder time‑step"); plt.ylabel("Decoder time‑step")
        plt.savefig(f"heatmaps/heatmap_{i}.png")
        plt.close()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # baseline run
    base_cfg = {"emb_dim":64,"hid_dim":64,"cell":"LSTM","dropout":0.0,
                "lr":1e-3,"bs":64,"epochs":5}
    vanilla = build_baseline(len(enc_map), len(dec_map),
                              base_cfg["emb_dim"], base_cfg["hid_dim"],
                              cell=base_cfg["cell"], dropout=base_cfg["dropout"])
    train_and_eval(vanilla, base_cfg, name="vanilla")

    # attention run
    attn = build_attention(len(enc_map), len(dec_map),
                            base_cfg["emb_dim"], base_cfg["hid_dim"],
                            cell=base_cfg["cell"], dropout=0.0)
    train_and_eval(attn, base_cfg, name="attention")

    # plot a few heatmaps
    plot_heatmaps(attn, num_examples=9)
    print(" Saved attention heatmaps under ./heatmaps/")  


