import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KAGGLE_API_TOKEN"] = "KGAT_ed1f53dba54456d5ab51a8d876c5b637"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

import tensorflow as tf
import keras
import keras_hub

import yfinance as yf

gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en", dtype="float16")

template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

prompt = template.format(
    instruction = "What is your favorite movie?",
    response = ""
)

gemma_lm.backbone.enable_lora(rank = 4)
sentiment_raw = pd.read_csv(os.path.join("sentiment_dataset", "all-data.csv"), encoding='latin1', header = None)
sentiment_raw.columns = ["Output", "Input"]
sentiment_dataset = sentiment_raw[["Input", "Output"]]
sentiment_dataset.insert(0, 'Instruction', 'You are a Quant Agent that generate sentiments for financial news headlines from the perspective of a retail investor. Your response should be either "neutral", "negative", or "positive", and nothing else.')

def create_prompt(row):
    return f"Instruction:\n{row['Instruction']}\n\nInput:\n{row['Input']}\n\nResponse:\n{row['Output']}"

sentiment_dataset['text'] = sentiment_dataset.apply(create_prompt, axis=1)

sentiment_dataset_combined = sentiment_dataset['text'].values

gemma_lm.preprocessor.sequence_length = 256
optimizer = keras.optimizers.AdamW(
    learning_rate = 5e-5,
    weight_decay = 0.01
)

optimizer.exclude_from_weight_decay(var_names = ["bias", "scale"])

gemma_lm.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = optimizer,
    weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()]
)

gemma_lm.fit(sentiment_dataset_combined[:10], epochs = 1, batch_size = 1)

template = "Instruction:\n{instruction}\n\nInput:{input}\n\nResponse:\n{response}"
prompt = template.format(
    instruction = "You are a Quant Agent that generate sentiments for financial news headlines from the perspective of a retail investor.",
    input = "UK proposes forcing Google to let publishers opt out of AI summaries",
    response = "[Your answer comes here]"
)
print("Start generating answer...")
output = gemma_lm.generate(prompt, max_length = 50)
print(output)