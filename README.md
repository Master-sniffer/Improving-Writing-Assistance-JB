# Formality Classification Project: A Student-Friendly Walkthrough

Dear Sir or Madam! This doc walks you through everything I did in my project on classifying text by formality level. I’ll cover what data I used, tools and libraries, how I built the model, the results, challenges, and how you can run it yourself.

---

## 1. Dataset

I used a dataset from Hugging Face called **oishooo/formality_classification**. It has short texts labeled as:
- Formal
- Informal
- Neutral

### How I processed the data:
- Loaded the dataset using Hugging Face's `datasets` library
- Used only the "train" split since that’s how it comes
- Each item has a `text` and a `formality_label`
- Shuffled the data and split it into:
  - 70% for training
  - 15% for validation
  - 15% for testing
- Tokenized the texts with a 128-token limit

```python
from datasets import load_dataset
import random

dataset = load_dataset("oishooo/formality_classification")
data = dataset["train"]

data_list = []
for item in data:
    text = str(item["text"]) if item["text"] else ""
    label = item["formality_label"]
    data_list.append({"text": text, "label_str": label})

random.shuffle(data_list)
```

---

## 2. Tools I Used

### Core Libraries:
- `PyTorch` for deep learning
- `Transformers` (Hugging Face) for the pre-trained model
- `Datasets` (Hugging Face) to load and handle data

### Model:
- **DistilBERT** – a smaller, faster version of BERT

### Extras:
- `scikit-learn`, `NumPy`, `Pandas` – for metrics, processing, and managing data
- `Optuna` – for tuning model hyperparameters
- `SHAP` – to understand why the model makes predictions
- `UMAP`, `Matplotlib`, `Plotly` – for data and model visualization

---

## 3. Metrics I Tracked

To evaluate the model, I used:
- **Accuracy** – overall correctness
- **Precision** – how often predictions were right
- **Recall** – how many actual labels we caught
- **F1 Score** – balance between precision and recall
- **Confidence Interval** – to show uncertainty in accuracy

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
```

---

## 4. How I Built It

### Step 1: Prepare the Data
- Loaded and tokenized the text
- Split it into train, validation, and test sets

### Step 2: Train the Model
- Used `distilbert-base-uncased` from Hugging Face
- Set up a `Trainer` from `transformers`
- Tuned parameters like learning rate and number of epochs using Optuna

```python
def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [2, 3, 4]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
    }
```

### Step 3: Cross-Validation
- Used Stratified K-Fold (k=3) to make sure each fold had all label types
- Trained and evaluated on each fold

### Step 4: Explain Results
- Used SHAP to see which words affected the prediction
- Used UMAP to visualize embeddings from the model

---

## 5. Results

### Accuracy with Confidence Intervals
Used Wilson score interval to calculate 95% confidence bounds:
```python
z = 1.96
center = (acc + (z**2)/(2*n)) / (1 + (z**2)/n)
margin = (z * np.sqrt((acc*(1-acc) + (z**2)/(4*n)) / n)) / (1 + (z**2)/n)
ci_lower = center - margin
ci_upper = center + margin
```

### Misclassifications
I also checked where the model made mistakes:
```python
mis_idx = np.where(pred_classes != labels_true)[0]
```

### Visuals
- Used UMAP and SHAP to make plots and understand the model behavior better

---

## 6. Challenges I Faced

1. **Hyperparameter Optimization**: Finding the optimal hyperparameters required extensive experimentation. The project used Optuna to systematically search the hyperparameter space, balancing between exploration and exploitation.

2. **Statistical Significance**: Ensuring the reliability of results required implementing confidence intervals for accuracy metrics, which helped quantify the uncertainty in model performance.

3. **Model Explainability**: Understanding why the model makes certain predictions was challenging. SHAP was used to provide insights into feature importance and model decision-making.

4. **Handling Text Data**: Processing and tokenizing text data with appropriate maximum length required careful consideration to balance between information preservation and computational efficiency.

5. **Cross-Validation Strategy**: Implementing stratified k-fold cross-validation was necessary to ensure balanced class distribution across folds, especially important for potentially imbalanced datasets.
---

## 7. How to Reproduce Everything

### Step 1: Set Up Your Environment
```bash
pip install datasets transformers scikit-learn shap umap-learn matplotlib optuna plotly
```
Make sure you also have PyTorch installed.

### Step 2: Load the Dataset
```python
from datasets import load_dataset
dataset = load_dataset("oishooo/formality_classification")
```

### Step 3: Tokenize and Prepare Data
Follow the process from the data preparation section above.

### Step 4: Initialize the Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label={0: "formal", 1: "informal", 2: "neutral"},
    label2id={"formal": 0, "informal": 1, "neutral": 2}
)
```

### Step 5: Training
Set training args:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./formality-clf-output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

Then train:
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### Step 6: Optimize Hyperparameters
```python
search_result = trainer.hyperparameter_search(
    backend="optuna",
    direction="minimize",
    hp_space=hp_space_optuna,
    n_trials=8,
    compute_objective=lambda metrics: metrics["eval_loss"]
)
```

### Step 7: Evaluate
```python
test_dataset = build_dataset(test_data_list, tokenizer)
test_results = trainer.evaluate(test_dataset)
```

Check misclassifications and compute confidence intervals as shown earlier.

### Step 8: Explain the Model
```python
import shap

explainer = shap.KernelExplainer(model_predict_for_shap, background_data)
shap_values = explainer.shap_values(test_texts)
```

---

