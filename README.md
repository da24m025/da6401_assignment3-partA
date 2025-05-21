

# Dakshina Transliteration (Assignment 3, Part A)
**DA24M025** 
[Connectivity Visualization Demo](https://drive.google.com/file/d/18BYq6F43r6F1PT6niaP5O-98PrkKZM4t/view?usp=sharing).
[Wandb Report](https://api.wandb.ai/links/da24m025-iit-madras/gxkvman2)
This repository contains code and results for a sequence‑to‑sequence transliteration task (Latin → Devanagari) using:

1. **Vanilla** RNN/GRU/LSTM seq2seq  
2. **Hyperparameter sweeps** with Weights & Biases  
3. **Attention‑augmented** seq2seq and heatmap visualizations  

All experiments are run on the Hindi subset of the Dakshina dataset (`hi.translit.sampled.*.tsv`).  

---

##  Repository Layout
````
├── Src/
│   ├── seq2seq\_vanilla\_transliteration\_bestparams\_test.py
│   ├── seq2seq\_transliteration\_sweep.py
│   ├── seq2seq\_attention\_sweep.py
│   └── attention\_seq2seq\_transliteration\_heatmaps.py
│
├── predictions\_vanilla/
│   └── \*.csv        ← entire test‑set outputs from best vanilla model
│
├── predictions\_attention/
│   └── \*.csv       ← entire test‑set outputs from best attention model
│
└── README.md
└── Connectivity Visualization.mp4 # Screen Recording of connectivity Visualizer
└── Connectivity Visualization Example # Connectivity of an example from the test data folder

````

---

##  Environment Setup

1. Clone this repository into your Kaggle notebook or local machine.
2. Install required Python packages:
   ```bash
   pip install tensorflow wandb matplotlib plotly


3. If you run in Kaggle, you don’t need to install CUDA or TensorFlow—just import.

---

##  How to Run

### 1. Vanilla seq2seq with Best Parameters

```bash
python Src/seq2seq_vanilla_transliteration_bestparams_test.py
```

* Reads the `hi.translit.sampled.*.tsv` files from `/kaggle/input/dakshina/...`.
* Builds a flexible encoder–decoder (RNN/GRU/LSTM) with your best hyperparameters.
* Evaluates on the test split, prints exact‐match accuracy.
* Saves full test‑set predictions under `predictions_vanilla/`.

---

### 2. Hyperparameter Sweep (Vanilla)

```bash
python Src/seq2seq_transliteration_sweep.py
```

* Defines a W\&B sweep (`bayes` method) over:

  * `emb_dim`: \[16, 32, 64, 128]
  * `enc_layers`: \[1, 2]
  * `dec_layers`: \[1]
  * `hid_size`: \[32, 64, 128]
  * `cell_type`: \[RNN, GRU, LSTM]
  * `dropout`/`rec_dropout`: \[0.0, 0.2, 0.4]
  * `learning_rate`: log‑uniform \[1e‑4, 1e‑2]
  * `batch_size`: \[32, 64]
* Logs training & validation loss/accuracy, sequence‐level exact match.
* After sweep, use the W\&B UI to identify best run and export its parameters.

---

### 3. Attention Augmented seq2seq Sweep

```bash
python Src/seq2seq_attention_sweep.py
```

* Single‐layer encoder & decoder + additive attention.
* Sweep config (bayes) over:

  * `emb_dim`: \[32, 64, 128]
  * `hid_size`: \[32, 64, 128]
  * `cell_type`: \[LSTM, GRU]
  * `dropout`: \[0.0, 0.2]
  * `learning_rate`: log‑uniform \[1e‑4, 1e‑3]
  * `batch_size`: \[32, 64]
  * `epochs`: 8
* Logs sequence‐level exact match on validation and test accuracy.
* Saves full test predictions under `predictions_attention/`.

---

### 4. Attention Heatmaps

```bash
python Src/attention_seq2seq_transliteration_heatmaps.py
```

* Reconstructs your best attention model.
* For the first *N* test examples, plots an encoder–decoder alignment heatmap.
* Outputs PNG files into `heatmaps/` directory.

---

##  Results

* **Best Vanilla Test Accuracy**: *29.2 %*
* **Best Attention Test Accuracy**: *37.2 %*
* Predictions for all test samples are in the respective `predictions_*` folders as `.csv` files.

---

##  Reporting & Visualizations

* All W\&B sweep dashboards are under:

  * `da24m025/assignment3` for vanilla
  * `da24m025/assignment3_attn` for attention
* Use the W\&B Reports editor to embed:

  ````markdown
  ```python
  # example: load and display heatmap
  img = plt.imread("heatmaps/heatmap_0.png")
  plt.imshow(img); plt.axis("off")
  ````

  ```
  ```

---

##  Observations 

1. **Model Comparison**

   * RNN vs. GRU vs. LSTM convergence speed and final accuracy
   * Effect of dropout and recurrent‐dropout

2. **Attention Benefits**

   * Improved handling of long‐range dependencies
   * Qualitative heatmaps illustrate which source chars influence each target char

3. **Challenge Extension**

   * Interactive Plotly “connectivity” visualization (Distill‑style)
   * Cursor‐hover bars showing per‑step attention weights

---

##  References

* Dakshina Dataset: [https://dataset.org/dakshina](https://dataset.org/dakshina)
* W\&B Keras Integration: [https://docs.wandb.ai/guides/integrations/keras](https://docs.wandb.ai/guides/integrations/keras)
* Attention visualization tutorial:
  – Medium: “Visualising LSTM activations in Keras”
  – Distill: “Memorization in RNNs”

---

