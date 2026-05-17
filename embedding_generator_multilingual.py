import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Embedding generator using device: {device}")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', dtype=torch.bfloat16).to(device).eval()

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

RETRIEVAL_TASK = 'Given a personal knowledge base query, retrieve relevant passages that answer the query'

def generate_embeddings(sentences_and_sources, is_query):
    # Queries use a task instruction prefix; passages are embedded as-is.
    sentences = []
    for sentence, source in sentences_and_sources:
        if is_query:
            sentences.append(f'Instruct: {RETRIEVAL_TASK}\nQuery: {sentence}')
        else:
            sentences.append(sentence)
    batch_dict = tokenizer(sentences, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.inference_mode():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.float().cpu().numpy().tolist()
