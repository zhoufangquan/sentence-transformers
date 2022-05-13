# %%
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
# %%
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

sentencese_embeddings = model.encode(sentences)
# %%
for S, S_embed in zip(sentences, sentencese_embeddings):
    print("Sentence:", S)
    print("Embedding:", S_embed)
    print("")

print(sentencese_embeddings.shape)

# %%
# 查看两个句子的相似度
emb1 = model.encode("This is a red cat with a hat.")
emb2 = model.encode("Have you seen my red cat?")
print(util.cos_sim(emb1, emb2))
# %%
