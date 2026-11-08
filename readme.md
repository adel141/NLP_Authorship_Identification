# Authorship Attribution with Classical NLP, BERT, RoBERTa and a Stacked Ensemble

**Natural Language Processing, Summer 2025-2026, Section D, Group 09**
American International University-Bangladesh (AIUB), Faculty of Science & Technology, Department of Computer Science

Who wrote this text? This project answers that question on two datasets of very different difficulty, and compares three families of models:

1. **Classical NLP** (midterm): Bag-of-Words and TF-IDF with Naive Bayes, Logistic Regression and Linear SVM.
2. **Transformers** (final): `bert-base-uncased` and `roberta-base`, each fine-tuned over an 8-setting hyperparameter grid.
3. **Stacked ensemble** (final): the best BERT and best RoBERTa combined by a Logistic Regression meta-learner.

## Team

| Member | ID | 
|---|---|
| Eva Jahan | 23-55536-3 |
| **Hamed Mohammed Adel** | 23-55566-3 | 

## Datasets

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Name | Reuters 50-50 (C50) news corpus | Spooky Author Identification |
| Authors (classes) | 50 | 3 (Edgar Allan Poe, H.P. Lovecraft, Mary Shelley) |
| Difficulty | Hard: many authors, about 70 training documents each | Easier: 3 classes, much more data per class |

Each dataset was split **70% train / 15% validation / 15% test**, stratified by author, with a fixed random seed.

## Method

**Preprocessing (shared by every model):** tokenization, case folding, lemmatization, stop-word removal and punctuation removal.

**Classical models:** Multinomial Naive Bayes, Logistic Regression and Linear SVM, each on Bag-of-Words and on TF-IDF features.

**Transformers:** BERT (`bert-base-uncased`) and RoBERTa (`roberta-base`) fine-tuned for sequence classification with the Hugging Face stack.

| Setting | Value |
|---|---|
| Max sequence length | 128 tokens |
| Epochs | up to 5, early stopping (patience 2) |
| Optimizer | AdamW, warmup ratio 0.1, dropout 0.1 |
| Learning rate | 2e-5, 3e-5 |
| Batch size | 16, 32 |
| Weight decay | 0.01, 0.1 |
| Grid | 2 x 2 x 2 = 8 configurations per model per dataset |
| Hardware | single Kaggle T4 GPU (final), Google Colab (midterm) |

**Stacked ensemble:** the softmax probability vectors of the best BERT and best RoBERTa configuration are concatenated, and a Logistic Regression meta-learner (`max_iter=1000`) is trained on them using the validation set. At test time the same concatenation is fed to the meta-learner.

## Results

Best configuration of every method. Accuracy, as reported in the project report.

| Model | Dataset 1 (50 authors) | Dataset 2 (3 authors) |
|---|---|---|
| Naive Bayes (BoW) | 65.44% | **83.68%** |
| Logistic Regression | 67.40% | 81.69% |
| Linear SVM | 66.96% | 81.92% |
| BERT | 68.13% | 81.10% |
| RoBERTa | **69.73%** | 79.02% |
| Stacked ensemble | 69.33% | 83.52% |

Best fine-tuning settings:

| | Dataset 1 | Dataset 2 |
|---|---|---|
| BERT | lr 3e-5, batch 16, wd 0.01 | lr 3e-5, batch 32, wd 0.1 |
| RoBERTa | lr 3e-5, batch 16, wd 0.1 | lr 3e-5, batch 16, wd 0.01 |

### Key findings

- **No single method wins everywhere.** On the hard 50-author task, transformers clearly beat the classical models: RoBERTa (69.73%) and the ensemble (69.33%) both pass the best classical result (67.40%).
- **On the easy 3-author task**, a simple Naive Bayes Bag-of-Words model (83.68%) is essentially tied with the ensemble (83.52%) and ahead of BERT and RoBERTa alone.
- **Stacking helps, but not always.** The ensemble improved over both base models on Dataset 2 (83.52% vs 81.10% and 79.02%), yet fell just short of RoBERTa alone on Dataset 1.
- **Smaller batches won.** Batch size 16 beat batch size 32 across the Dataset 1 grid for both transformers.
- **Complexity pays off when the task is hard and data per class is scarce**, not automatically for easier problems.

## Limitations

1. The lowercased, lemmatized, stop-word-stripped text used for the classical models was also fed to BERT and RoBERTa, which removes the casing and punctuation cues transformers use. This likely understates their performance.
2. Only 8 hyperparameter combinations per model were tried, and larger models (`bert-large`, `roberta-large`) were not explored.
3. Results come from a single stratified split, not k-fold cross-validation. Dataset 1 has only about 15 validation and 15 test documents per author, so the numbers carry sampling variance.
4. The ensemble uses only two base models and a plain Logistic Regression meta-learner.
5. Inputs were truncated to 128 tokens, which can drop signal from longer Reuters articles.
6. Training ran on one T4 GPU for at most 5 epochs, which may under-train some configurations.

## Tech stack

Python, pandas, scikit-learn, NLTK, PyTorch, Hugging Face Transformers and Datasets, Jupyter notebooks (Google Colab and Kaggle).

## Reproducing

1. Download the Reuters 50-50 (C50) corpus and the Spooky Author Identification dataset.
2. Run the preprocessing and the classical baselines (midterm notebook).
3. Fine-tune BERT and RoBERTa over the 8-setting grid on each dataset (final notebook).
4. Train the stacking meta-learner on the best BERT and RoBERTa validation probabilities and evaluate on the test split.

## Documents

- `Midterm report` (classical pipeline and results)
- `Final-term report` (transformers, ensemble and analysis)