import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dvc.api import params_show

#model_name = "BAAI/bge-m3"  # Replace with your model
# Load parameters from params.yaml
params = params_show()
model_name = params["general"]["embedding_model"]
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the GPU
model = model.to(device)
model.eval()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]  # CLS token is the first token

def generate_embeddings(texts, batch_size=100):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            norm = torch.norm(cls_pooling(outputs))
            embeddings.append(cls_pooling(outputs)/norm)
    return torch.cat(embeddings)