import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dvc.api import params_show
import numpy as np

# Load parameters from params.yaml
params = params_show()
model_name = params["general"]["embedding_model"]
embedding_dim = params["general"]["embedding_dim"]
batch_size = params["general"]["batch_size"]

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU and wrap it with DataParallel
model = model.to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = torch.nn.DataParallel(model)
model.eval()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]  # CLS token is the first token

def generate_embeddings(texts, batch_size=batch_size, task="text-matching"):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            # inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model.encode(batch_texts,  truncate_dim=embedding_dim, task="text-matching")
            # pooled_output = cls_pooling(outputs)
            # norm = torch.norm(pooled_output, dim=1, keepdim=True)
            # normalized_embeddings = pooled_output / norm
            embeddings.append(outputs)
    return np.concatenate(embeddings)

