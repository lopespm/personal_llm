import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings(sentences_and_sources, is_query):
    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    sentences = []
    for sentence, source in sentences_and_sources:
        prefix = "query: " if is_query else "passage:"
        sentences.append(f'{prefix} {sentence}')
    batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().tolist()
