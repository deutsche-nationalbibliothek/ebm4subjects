import numpy as np
import torch
from dvc.api import params_show
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

params = params_show()
model_name = params["general"]["embedding_model"]
embedding_dim = params["general"]["embedding_dim"]
batch_size = params["general"]["batch_size"]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()


def generate_embeddings(
    texts: list[str], batch_size: int = batch_size, task: str = "text-matching"
):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            embeddings.append(
                model.encode(
                    texts[i : i + batch_size], truncate_dim=embedding_dim, task=task
                )
            )
    return np.concatenate(embeddings)
