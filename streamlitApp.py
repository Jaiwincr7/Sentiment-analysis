import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.utils import pad_sequences
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import torch

# --- Set wide page layout ---
st.set_page_config(layout="wide")

# --- Sentiment mapping for 5-class LSTM ---
sentiment_map = {
    1: "Very Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Very Positive"
}

# --- Cache model loading ---
@st.cache_resource
def load_lstm_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_bert_model(model_name):
    tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
    model_bert = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    return tokenizer_bert, model_bert

# --- Load models ---
lstm_model = load_lstm_model("sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
distilbert_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_bert, model_bert = load_bert_model(model_name)

# --- Initialize LIME explainer ---
class_names = [sentiment_map[i] for i in range(1, 6)]
explainer = LimeTextExplainer(class_names=class_names)

# --- Helper functions ---
@st.cache_data(show_spinner=False)
def cached_lstm_predict(seq):
    return lstm_model.predict(seq)

@st.cache_data(show_spinner=False)
def cached_distilbert_predict(text):
    return distilbert_pipe(text)[0]

@st.cache_data(show_spinner=False)
def cached_attention_heatmap(text):
    inputs = tokenizer_bert(text, return_tensors="pt")
    outputs = model_bert(**inputs)
    last_layer_attention = outputs.attentions[-1]
    attn = last_layer_attention.mean(dim=1)[0]
    cls_attn = attn[0].detach().numpy()
    tokens = tokenizer_bert.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, cls_attn

def lstm_predict_proba(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, padding='post', maxlen=100)
    preds = cached_lstm_predict(seqs)
    return preds

def visualize_attention(tokens, attn_scores):
    attn_norm = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-10)
    token_html = ""
    for token, score in zip(tokens, attn_norm):
        color = int(255 * (1 - score))  # higher attention = darker red
        token_html += f"<span style='background-color: rgb(255,{color},{color}); padding:2px; margin:1px;'>{token}</span> "
    return f"<div style='line-height:2'>{token_html}</div>"

# --- Streamlit UI ---
st.title("Custom LSTM vs DistilBERT")
st.markdown("- Enables real-time sentiment analysis and interpretability for user-provided text.")
st.markdown("- Tech Stack:- Python, TensorFlow/Keras, PyTorch, Transformers, LIME, Streamlit, NLTK, SKlearn, Pandas, Numpy")
st.markdown("- Compare predictions and interpretability between your LSTM model and a pre-trained DistilBERT model.")
st.markdown("- Trained a LSTM model from scratch with accuracy of *~65.4%* on the SST-5 dataset (~120k+ samples). DistilBert performed accuracy of *~91.3%* on SST-2")
st.markdown("- Preprocessing included stopword removal (NLTK), tokenization, and padding. Dataset split: 80-20 train-test with 10% validation.")
url = "https://colab.research.google.com/drive/1MAJqgH_nGKivf5RVDu3OZCtxTsrJoajP?usp=sharing"
st.markdown("- Check out the Colab notebook for preprocessing and training of the Model [Colab](%s)" % url)

text_input = st.text_area("Enter text:", "This movie was amazing!")

if st.button("Analyze"):
    # --- LSTM Prediction ---
    seq = tokenizer.texts_to_sequences([text_input])
    seq = pad_sequences(seq, padding='post', maxlen=100)
    lstm_pred = cached_lstm_predict(seq)
    lstm_class = int(lstm_pred.argmax(axis=1)[0]) + 1
    confidence = float(np.max(lstm_pred[0]))

    # --- DistilBERT Prediction ---
    distilbert_result = cached_distilbert_predict(text_input)

    # --- Side-by-side predictions ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔹 Custom LSTM Prediction")
        st.markdown(f"**Predicted Sentiment:** {sentiment_map[lstm_class]} ({lstm_class})")
        st.markdown(f"**Confidence:** {confidence:.2f}")
    with col2:
        st.subheader("🔹 DistilBERT Prediction")
        st.markdown(f"**Label:** {distilbert_result['label']}")
        st.markdown(f"**Score:** {distilbert_result['score']:.2f}")

    st.markdown("---")

    # --- LIME + Attention side-by-side ---
    col1, col2 = st.columns(2, gap="large")

    # LIME Explanation
    with col1:
        st.subheader("🧠 LIME Explanation (LSTM)")
        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_instance(
                text_input,
                lstm_predict_proba,
                num_features=10,
                labels=[lstm_class - 1]
            )
            lime_html = exp.as_html()
            custom_title = f"<h3 style='color:#333'>Predicted Sentiment: {sentiment_map[lstm_class]} ({lstm_class})</h3>"
            lime_html = f"""
            <div style="background-color: white; padding: 15px; border-radius: 12px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.2);">
                {custom_title}
                {lime_html}
            </div>
            """
            components.html(lime_html, height=500, scrolling=True)

    # Attention Heatmap
    with col2:
        st.subheader("🧠 Attention Heatmap (DistilBERT)")
        tokens, attn_scores = cached_attention_heatmap(text_input)
        attn_html = visualize_attention(tokens, attn_scores)
        components.html(attn_html, height=500, scrolling=True)
