import os
import sys
import warnings

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import keras_hub

import keras
from keras import layers, models

from src.config import config

from pprint import pprint
import nlpaug.augmenter.word as naw
from pygooglenews import GoogleNews

import nltk
from nltk.corpus import wordnet
import builtins

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

PRETRAINED_MODEL_WEIGHT_PATH = os.path.join(BASE_DIR, "models", "weights", "model_4L_weights_cp_best.weights.h5")
MODEL_WEIGHT_PATH = os.path.join(BASE_DIR, "models", "weights", "march_sixth_4Layers.weights.h5")

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\\^_`{|}~"), ""
    )

def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    vectorize_layer = layers.TextVectorization(
        max_tokens = vocab_size,
        output_mode = "int",
        standardize = custom_standardization,
        output_sequence_length = max_seq
    )
    vectorize_layer.adapt(texts)

    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2: vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer

NEWS_TEXT_RAW = pd.read_csv(os.path.join(BASE_DIR, "data", "abcnews-date-text.csv"))["headline_text"]

VECTORIZE_LAYER = get_vectorize_layer(
    NEWS_TEXT_RAW.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    special_tokens=["[mask]"],
)

ID2TOKEN = dict(enumerate(VECTORIZE_LAYER.get_vocabulary()))
TOKEN2ID = {y: x for x, y in ID2TOKEN.items()}
MASK_TOKEN_ID = VECTORIZE_LAYER(["[mask]"]).numpy()[0][0]
SAMPLE_TOKENS = VECTORIZE_LAYER(["Google wont [mask] replacing our news headlines with terrible AI"])


def encode(texts, vectorize_layer = VECTORIZE_LAYER):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()

def get_masked_input_and_labels(encoded_texts, mask_token_id=MASK_TOKEN_ID):
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    inp_mask[encoded_texts <= 2] = False
    labels = -1 * np.ones(encoded_texts.shape, dtype = int)
    labels[inp_mask] = encoded_texts[inp_mask]

    encoded_texts_masked = np.copy(encoded_texts)
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[inp_mask_2mask] = (mask_token_id)

    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1/9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(3, mask_token_id, inp_mask_2random.sum())

    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights

def get_mlm_dataset():
    news_raw = pd.read_csv(os.path.join("data", "abcnews-date-text.csv"))
    news_text_raw = news_raw["headline_text"]

    vectorize_layer = get_vectorize_layer(
        news_text_raw.tolist(),
        config.VOCAB_SIZE,
        config.MAX_LEN,
        special_tokens=["[mask]"],
    )

    mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]

    x_all_encoded = encode(news_text_raw)

    x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(x_all_encoded)

    mlm_ds = tf.data.Dataset.from_tensor_slices(
        (x_masked_train, y_masked_labels, sample_weights)
    )
    mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
    mlm_ds_small = mlm_ds.shard(num_shards=256, index=0)

current_dir = os.path.dirname(os.path.abspath(__file__))
venv_data_path = os.path.join(current_dir, "..", "..", ".venv", "nltk_data")
os.makedirs(venv_data_path, exist_ok=True)

if venv_data_path not in nltk.data.path:
    nltk.data.path.insert(0, venv_data_path)

nltk.download('wordnet', download_dir=venv_data_path)
nltk.download('omw-1.4', download_dir=venv_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=venv_data_path)
nltk.download('punkt_tab', download_dir=venv_data_path)

builtins.wordnet = wordnet

AUG = naw.SynonymAug(aug_src='wordnet')

def augment_text(df, target_count, aug=AUG):
    current_count = len(df)
    if current_count >= target_count:
        return df
    
    aug_samples = []
    needed = target_count - current_count
    label_name = df.iloc[0]['Output']
    print(f"Augmenting '{label_name}' class: {current_count} -> {target_count}...")
    
    while len(aug_samples) < needed:
        for text in df['Input']:
            if len(aug_samples) >= needed:
                break
            augmented_text = aug.augment(text)[0]
            aug_samples.append(augmented_text)
            
    df_aug = pd.DataFrame({'Input': aug_samples, 'Output': [label_name] * len(aug_samples)})
    return pd.concat([df, df_aug])

def initialize_le():
    sentiment_raw = pd.read_csv(os.path.join(BASE_DIR, "data", "sentiment.csv"), encoding='latin1', header=None)
    sentiment_raw.columns = ["Output", "Input"]

    df_neg = sentiment_raw[sentiment_raw["Output"] == "negative"]
    df_neu = sentiment_raw[sentiment_raw["Output"] == "neutral"]
    df_pos = sentiment_raw[sentiment_raw["Output"] == "positive"]

    aug = naw.SynonymAug(aug_src='wordnet')
    target_n = 2000 
    df_neg_final = augment_text(df_neg, target_n, aug=aug)
    df_neu_final = augment_text(df_neu, target_n, aug=aug)
    df_pos_final = augment_text(df_pos, target_n, aug=aug)

    final_df = pd.concat([df_neg_final, df_neu_final, df_pos_final]).sample(frac=1, random_state=42).reset_index(drop=True)

    le = LabelEncoder()
    le.fit_transform(final_df['Output'])
    return le


def bert_module(query, key, value, i, mask=None):
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}_multiheadattention".format(i),
    )(query, key, value, attention_mask=mask)
    attention_output = layers.Dropout(0.1, name="encoder_{}_att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_att_layernormalization".format(i)
    )(query + attention_output)

    ffn = models.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}_ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}_ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output

LOSS_FN = keras.losses.SparseCategoricalCrossentropy(reduction=None)
LOSS_TRACKER = keras.metrics.Mean(name="loss")

class MaskedLanguageModel(keras.Model):

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, loss_fn=LOSS_FN, loss_tracker=LOSS_TRACKER):
        loss = loss_fn(y, y_pred, sample_weight)
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        return keras.ops.sum(loss)

    def compute_metrics(self, x, y, y_pred, sample_weight, loss_tracker=LOSS_TRACKER):
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self, loss_tracker=LOSS_TRACKER):
        return [loss_tracker]
    
def create_masked_language_bert_model(vectorize_layer=VECTORIZE_LAYER):
    inputs = layers.Input(shape=(config.MAX_LEN,), name="Input")
    
    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, 
        config.EMBED_DIM, 
        mask_zero=True, 
        name="word_embedding"
    )(inputs)
    position_embeddings = keras_hub.layers.PositionEmbedding(
        sequence_length=config.MAX_LEN,
        name="position_embedding"
    )(word_embeddings)
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(1, 4+1):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model

class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens, id2token=ID2TOKEN):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id, id2token=ID2TOKEN):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None, mask_token_id = MASK_TOKEN_ID, sample_tokens=SAMPLE_TOKENS):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)

class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens, id2token=ID2TOKEN):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id, id2token=ID2TOKEN):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None, mask_token_id=MASK_TOKEN_ID, sample_tokens=SAMPLE_TOKENS):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)

def create_classifier_bert_model(pretrained_bert_model):
    inputs = layers.Input((config.MAX_LEN,), dtype="int64")
    sequence_output = pretrained_bert_model(inputs)
    
    pooled_output = layers.Lambda(lambda x: x[:, 0, :])(sequence_output)
    x = layers.BatchNormalization()(pooled_output)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(3, activation="softmax")(x)

    classifier_model = models.Model(inputs, outputs, name="classification")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5) 
    
    classifier_model.compile(
        optimizer=optimizer, 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return classifier_model

def predict(text, model, le):
    text_encoded = encode(text)
    # print("Encoded Text: ", text_encoded)
    
    pred = model(text_encoded, training=False)

    probabilities = tf.nn.softmax(pred, axis=-1).numpy()[0]

    pred_class = np.argmax(probabilities)
    sentiment = le.inverse_transform([pred_class])[0]
    confidence_score = probabilities[pred_class]

    class_names = le.classes_
    all_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return sentiment, confidence_score, all_probs

def predict_and_print(text, model):
    new_texts = [text]
    result, confidence, all_probs = predict(new_texts, model)
    
    print(f"Input Headline: \"{text}\"")
    print(f"Top Prediction: {result} ({confidence * 100:.2f}%)")
    print("-" * 30)
    print("Class Probabilities:")
    
    sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    
    for label, prob in sorted_probs:
        bar = "#" * int(prob * 20) 
        print(f" - {label:<12}: {prob * 100:>6.2f}% {bar}")

    return result

def save_model_weights(model, file_name, folder_name):
    os.makedirs(folder_name, exist_ok=True)

    words = file_name.split(".")

    model_name = words[0]

    existing_files = [f for f in os.listdir(folder_name) if f.startswith(model_name)]
    next_number = len(existing_files) + 1
    words[0] = f"{model_name}_{next_number}"

    file_name = ".".join(words)
    save_path = os.path.join(folder_name, file_name)

    model.save_weights(save_path)
    print(f"model saved to: {save_path}")

def load_SA_model(vectorize_layer=VECTORIZE_LAYER):
    '''
    Loads Sentiment Analysis Model.

    param:
        vectorize_layer : need to load dataset first, then a vectorized_layer 
                          that was created for it. Default to VECOTRIZE_LAYER
    '''

    bert_masked_model = create_masked_language_bert_model(vectorize_layer)
    bert_masked_model.load_weights(PRETRAINED_MODEL_WEIGHT_PATH)
    pretrained_bert_model = keras.Model(
        bert_masked_model.input, bert_masked_model.get_layer("encoder_4_ffn_layernormalization").output
    )

    classifier_model = create_classifier_bert_model(pretrained_bert_model)
    classifier_model.load_weights(MODEL_WEIGHT_PATH)
    return classifier_model

GN = GoogleNews(lang='en', country='US')
def get_news_for_prediction(ticker, gn=GN):
    search_query = f'stock:{ticker}'
    google_news = gn.search(search_query, when='1d')

    return google_news['entries']