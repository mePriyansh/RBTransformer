<h1 align="center">RBTransformer</h1>

<p align="center"><strong>Official PyTorch codebase of RBTransformer from our paper:<br>
<em>“A Brain Wave Encodes a Thousand Tokens: Modeling Inter-Cortical Neural Interactions for Effective EEG-Based Emotion Recognition”</em></strong></p>

## 1. *OVERVIEW*
<p align="center">
  <img src="assets/model-diagram.png" alt="RBTransformer Architecture" width="100%">
</p>
<p align="center"><b>Figure 1.</b> Model Architecture of RBTransformer</p>


## 2. *INSTALLATION AND SETUP*

### I. Installation

To get started, first clone the repository from GitHub:

```bash
git clone https://github.com/nnilayy/RBTransformer.git
```

Then navigate into the project folder and install dependencies:

```bash
cd RBTransformer
pip install -r requirements.txt
```

> The codebase requires **Python 3.10 or higher**.

To ensure scripts run correctly across environments, set the `PYTHONPATH` to the root of the project:

* **macOS / Linux (Terminal):**

  ```bash
  export PYTHONPATH=$(pwd)
  ```

* **Windows (PowerShell):**

  ```powershell
  $env:PYTHONPATH = (Get-Location).Path
  ```

* **Python Notebooks (Colab, Kaggle, Jupyter):**

  ```python
  import os
  os.environ['PYTHONPATH'] = os.getcwd()
  ```

---

### II. Preprocessing Datasets

RBTransformer is evaluated on three benchmark EEG datasets: **SEED**, **DEAP**, and **DREAMER**, along their respective affective dimensions, for both **Binary** and **Multi-Class Classification** tasks.

<p align="center">
  <img src="assets/eeg-preprocessing.png" alt="Preprocessing Pipeline" width="100%">
</p>
<p align="center"><b>Figure 2.</b> Preprocessing Pipeline for RBTransformer</p>

As illustrated in **Figure 2**, the preprocessing pipeline handles the full data transformation—from raw EEG signals to baseline-corrected BDE tokens—tailored for all **13 prediction tasks**. Once processed, the datasets are saved as `.pkl` files inside the `preprocessed_datasets/` directory and are used directly during training. The table below summarizes all 13 preprocessed dataset files, categorized by Dataset, Dimension, and Task Type:

| #  | Dataset | Task Type                  | Dimension | Output File                            |
|----|---------|----------------------------|-----------|----------------------------------------|
| 1  | SEED    | Multi-Class Classification | Emotion   | `seed_multi_emotion_dataset.pkl`       |
| 2  | DEAP    | Multi-Class Classification | Valence   | `deap_multi_valence_dataset.pkl`       |
| 3  | DEAP    | Multi-Class Classification | Arousal   | `deap_multi_arousal_dataset.pkl`       |
| 4  | DEAP    | Multi-Class Classification | Dominance | `deap_multi_dominance_dataset.pkl`     |
| 5  | DEAP    | Binary Classification      | Valence   | `deap_binary_valence_dataset.pkl`      |
| 6  | DEAP    | Binary Classification      | Arousal   | `deap_binary_arousal_dataset.pkl`      |
| 7  | DEAP    | Binary Classification      | Dominance | `deap_binary_dominance_dataset.pkl`    |
| 8  | DREAMER | Multi-Class Classification | Valence   | `dreamer_multi_valence_dataset.pkl`    |
| 9  | DREAMER | Multi-Class Classification | Arousal   | `dreamer_multi_arousal_dataset.pkl`    |
| 10 | DREAMER | Multi-Class Classification | Dominance | `dreamer_multi_dominance_dataset.pkl`  |
| 11 | DREAMER | Binary Classification      | Valence   | `dreamer_binary_valence_dataset.pkl`   |
| 12 | DREAMER | Binary Classification      | Arousal   | `dreamer_binary_arousal_dataset.pkl`   |
| 13 | DREAMER | Binary Classification      | Dominance | `dreamer_binary_dominance_dataset.pkl` |

To generate all 13 preprocessed datasets, simply run the following scripts:

```bash
python dataset_classes/deap_preprocessing.py
python dataset_classes/dreamer_preprocessing.py
python dataset_classes/seed_preprocessing.py
```

---

### III. Training Scripts

RBTransformer was evaluated on the SEED, DEAP, and DREAMER datasets across their affective dimensions, **emotion** for SEED, and **valence**, **arousal**, **dominance** for DEAP and DREAMER, for both Binary and Multi-Class classification settings.

To train and replicate the results of RBTransformer as indicated in the paper, run the following script with the given configuration options:

```bash
python train.py \
  --root_dir <ROOT_DIR> \
  --dataset <DATASET> \
  --task_type <TASK_TYPE> \
  --dimension <DIMENSION> \
  --seed 23 \
  --device cuda \
  --num_workers 4 \
  --hf_username <HF_USERNAME> \
  --hf_token <HF_TOKEN> \
  --wandb_api_key <WANDB_API_KEY>
```

Replace the following:

* `<ROOT_DIR>`: Path to the folder containing the preprocessed `.pkl` file. Default: `preprocessed_datasets`

* `<DATASET>`: Name of the dataset to train on. Options: `seed`, `deap`, `dreamer`

* `<TASK_TYPE>`: Type of classification task. Options: `binary`, `multi`

* `<DIMENSION>`: Affective dimension to predict.

  * For `seed`: `emotion`
  * For `deap` or `dreamer`: `valence`, `arousal`, `dominance`

* `<HF_USERNAME>`: Your Hugging Face username (used for model uploads)

* `<HF_TOKEN>`: Your Hugging Face access token (needed to authenticate and push models)

* `<WANDB_API_KEY>`: Your Weights & Biases API key (for logging training metrics)

---

### IV. Ablation Scripts

RBTransformer was ablated on the DREAMER dataset along the Arousal dimension for Binary Classification tasks.

To replicate these ablation experiments, use the scripts provided under the `ablations/` directory. Each folder corresponds to a specific ablation setting.

```bash
python ablations/<ABLATION_NAME>/train.py \
  --root_dir <ROOT_DIR> \
  --seed 23 \
  --device cuda \
  --num_workers 4 \
  --hf_username <HF_USERNAME> \
  --hf_token <HF_TOKEN> \
  --wandb_api_key <WANDB_API_KEY>
```

Replace the following:

* `<ABLATION_NAME>`: Name of the ablation to run. Choose one of:

  * `with-adasyn`
  * `without-dropout`
  * `without-electrode-identity-embedding`
  * `without-intercortical-attention`
  * `without-smote-and-label-smoothing`
  * `without-weight-decay`

* `<ROOT_DIR>`: Path to the folder containing the preprocessed DREAMER binary arousal `.pkl` file. By default, this is `preprocessed_datasets/`.

* `<HF_USERNAME>`: Your Hugging Face username (used for model uploads)

* `<HF_TOKEN>`: Your Hugging Face access token (needed to authenticate and push models)

* `<WANDB_API_KEY>`: Your Weights & Biases API key (for logging training metrics)

---

## 3. *W&B: EXPERIMENT LOGS*

All training runs were tracked using Weights & Biases.  
Each link below points to a grouped dashboard containing all 5 fold-specific logs per configuration.  
All key metrics such as **Validation Accuracy**, **F1-Score**, **Precision**, **Recall**, **Training Accuracy**, and **Training Loss** are tracked in detail.

--- 

### I. Training Logs

| #  | Dataset | Task Type                   | Dimension | W\&B Project Link                                                                                                                                       |
| -- | ------- | --------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | SEED    | Multi-Class Classification  | Emotion   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-seed-multi-emotion-class-classification-sota-run-0001?nw=nwusernnilayy)       |
| 2  | DEAP    | Multi-Class Classification  | Valence   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-multi-valence-class-classification-sota-run-0001?nw=nwusernnilayy)       |
| 3  | DEAP    | Multi-Class Classification  | Arousal   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-multi-arousal-class-classification-sota-run-0001?nw=nwusernnilayy)       |
| 4  | DEAP    | Multi-Class Classification  | Dominance | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-multi-dominance-class-classification-sota-run-0001?nw=nwusernnilayy)     |
| 5  | DEAP    | Binary-Class Classification | Valence   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-binary-valence-class-classification-sota-run-0001?nw=nwusernnilayy)      |
| 6  | DEAP    | Binary-Class Classification | Arousal   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-binary-arousal-class-classification-sota-run-0001?nw=nwusernnilayy)      |
| 7  | DEAP    | Binary-Class Classification | Dominance | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-deap-binary-dominance-class-classification-sota-run-0001?nw=nwusernnilayy)    |
| 8  | DREAMER | Multi-Class Classification  | Valence   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-multi-valence-class-classification-sota-run-0001?nw=nwusernnilayy)    |
| 9  | DREAMER | Multi-Class Classification  | Arousal   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-multi-arousal-class-classification-sota-run-0001?nw=nwusernnilayy)    |
| 10 | DREAMER | Multi-Class Classification  | Dominance | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-multi-dominance-class-classification-sota-run-0001?nw=nwusernnilayy)  |
| 11 | DREAMER | Binary-Class Classification | Valence   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-valence-class-classification-sota-run-0001?nw=nwusernnilayy)   |
| 12 | DREAMER | Binary-Class Classification | Arousal   | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-sota-run-0001?nw=nwusernnilayy)   |
| 13 | DREAMER | Binary-Class Classification | Dominance | [View Logs](https://wandb.ai/nnilayy/%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-dominance-class-classification-sota-run-0001?nw=nwusernnilayy) |

---

### II. Ablation Logs

| # | Dataset | Task Type                   | Dimension | Ablation                             | W\&B Project Link                                                                                                                                                                     |
| - | ------- | --------------------------- | --------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | DREAMER | Binary-Class Classification | Arousal   | Without Inter-Cortical Attention     | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-without-intercortical-attention-0001?nw=nwusernnilayy)      |
| 2 | DREAMER | Binary-Class Classification | Arousal   | Without Weight Decay                 | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-without-weight-decay-0001?nw=nwusernnilayy)                 |
| 3 | DREAMER | Binary-Class Classification | Arousal   | Without SMOTE & Label Smoothing      | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-without-smote-and-label-smoothing-0001?nw=nwusernnilayy)    |
| 4 | DREAMER | Binary-Class Classification | Arousal   | Without Dropout                      | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-without-dropout-0001?nw=nwusernnilayy)                      |
| 5 | DREAMER | Binary-Class Classification | Arousal   | Without Electrode Identity Embedding | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-without-electrode-identity-embedding-0001?nw=nwusernnilayy) |
| 6 | DREAMER | Binary-Class Classification | Arousal   | With ADASYN                          | [View Logs](https://wandb.ai/nnilayy/abl-%CE%B1-rbtransformer-eeg-recognition-dreamer-binary-arousal-class-classification-with-adasyn-0001?nw=nwusernnilayy)                          |

---

## 4. *HUGGING FACE: MODEL CHECKPOINTS*

The following tables list all pretrained RBTransformer models trained using 5-fold subject-dependent cross-validation.  
Each link points to a Hugging Face **collection** containing all 5 fold-specific checkpoints.

### I. Trained Model Checkpoints

| #  | Dataset | Task Type                   | Dimension | Hugging Face Collection Link                                                                                                             |
| -- | ------- | --------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | SEED    | Multi-Class Classification  | Emotion   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-seed-multi-class-classification-weights-685405050ad35ac2317bb718) |
| 2  | DEAP    | Multi-Class Classification  | Valence   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-multi-valence-weights-68540512f2302718187e6b73)              |
| 3  | DEAP    | Multi-Class Classification  | Arousal   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-multi-arousal-weights-68846a68bff5773cc32c0cbc)              |
| 4  | DEAP    | Multi-Class Classification  | Dominance | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-multi-dominance-weights-68846a7c5087ab57d5f49746)            |
| 5  | DEAP    | Binary-Class Classification | Valence   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-binary-valence-weights-68540522f92909b9c5fa1bab)             |
| 6  | DEAP    | Binary-Class Classification | Arousal   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-binary-arousal-weights-68846aa0f6d5858363d80e9e)             |
| 7  | DEAP    | Binary-Class Classification | Dominance | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-deap-binary-dominance-weights-68846a94a4a33706a17c37f6)           |
| 8  | DREAMER | Multi-Class Classification  | Valence   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-multi-valence-weights-68846b1dce02622a9167336a)           |
| 9  | DREAMER | Multi-Class Classification  | Arousal   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-multi-arousal-weights-685405310a8788a07fac2357)           |
| 10 | DREAMER | Multi-Class Classification  | Dominance | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-multi-dominance-weights-68846cfdce02622a916787df)         |
| 11 | DREAMER | Binary-Class Classification | Valence   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-binary-valence-weights-68846af11784581c5aca0999)          |
| 12 | DREAMER | Binary-Class Classification | Arousal   | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-binary-arousal-weights-6854053ec1a6746032c2fc18)          |
| 13 | DREAMER | Binary-Class Classification | Dominance | [View Models](https://huggingface.co/collections/nnilayy/rbtransformer-dreamer-binary-dominance-weights-68846af96b4462ce01d72be9)        |

---

### II. Ablation Model Checkpoints

| # | Dataset | Task Type                   | Dimension | Ablation                             | Hugging Face Collection Link                                                                                                                 |
| - | ------- | --------------------------- | --------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | DREAMER | Binary-Class Classification | Arousal   | Without Inter-Cortical Attention     | [View Models](https://huggingface.co/collections/nnilayy/abl-w-o-ic-attention-rbtransformer-dreamer-binary-arousal-68860757036423affcdcb951) |
| 2 | DREAMER | Binary-Class Classification | Arousal   | Without Weight Decay                 | [View Models](https://huggingface.co/collections/nnilayy/abl-w-o-weight-decay-rbtransformer-dreamer-binary-arousal-6886068d12825c3fe6347b8f) |
| 3 | DREAMER | Binary-Class Classification | Arousal   | Without SMOTE & Label Smoothing      | [View Models](https://huggingface.co/collections/nnilayy/abl-w-o-smote-and-ls-rbtransformer-dreamer-binary-arousal-688606fbe630a45168f0d1e3) |
| 4 | DREAMER | Binary-Class Classification | Arousal   | Without Dropout                      | [View Models](https://huggingface.co/collections/nnilayy/abl-w-o-dropout-rbtransformer-dreamer-binary-arousal-68860644ede5d03681f341b7)      |
| 5 | DREAMER | Binary-Class Classification | Arousal   | Without Electrode Identity Embedding | [View Models](https://huggingface.co/collections/nnilayy/abl-w-o-eie-rbtransformer-dreamer-binary-arousal-68860734f0f7573b8ed81c87)          |
| 6 | DREAMER | Binary-Class Classification | Arousal   | With ADASYN                          | [View Models](https://huggingface.co/collections/nnilayy/abl-with-adasyn-rbtransformer-dreamer-binary-arousal-688605f7581ba118ab5ff380)      |

---

## 5. *CITATION*

This work is currently **under peer review**.  
A citation will be added here as soon as the paper is accepted or published.


## 6. *VISUALIZATIONS*

To aid interpretability, we visualize both the learned feature representations and the model’s prediction behavior using t-SNE and confusion matrices.

<p align="center">
  <img src="assets/tsne-plots-rbtransformer.png" alt="t-SNE Plot" width="100%">
</p>
<p align="center"><b>Figure 3a.</b> t-SNE visualization of RBTransformer’s latent feature space showing clear emotion-based clustering.</p>

<p align="center">
  <img src="assets/confusion-matrices.png" alt="Confusion Matrix" width="100%">
</p>
<p align="center"><b>Figure 3b.</b> Confusion matrix illustrating prediction breakdown and misclassification patterns across emotion classes.</p>


