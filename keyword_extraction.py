from keybert import KeyBERT
from transformers import RobertaModel, RobertaTokenizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, HDBSCAN
from os import walk
import numpy as np
import csv

def collect_manual_tags():
    with open('data/clean_quoted_dataset.csv', newline = '', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        index = 0
        tags = []
        for row in spamreader:
            if index >= 1:
                # print(str(index) + " " + row[5])
                for tag in row[5].split(", "):
                    if tag not in tags:
                        tags.append(tag)
            index = index + 1
        return tags
    
def collect_manual_descriptions():
    with open('data/clean_quoted_dataset.csv', newline = '', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        index = 0
        descriptions = {}
        for row in spamreader:
            if index >= 1:
                # print(str(index) + " " + row[5])
                descriptions[int(row[0])] = "[" + row[0] + "]" + " " + row[1] + ": " + row[5]
            index = index + 1
        return descriptions
    
def collect_titles():
    with open('data/clean_quoted_dataset.csv', newline = '', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        index = 0
        titles = {}
        for row in spamreader:
            if index >= 1:
                # print(str(index) + " " + row[5])
                titles[int(row[0])] = "[" + row[0] + "]" + " " + row[1]
            index = index + 1
        return titles
    
def generate_tags_one_webpage(webpage_text_filename, seed_keywords):
    text_file = open(webpage_text_filename, 'r', encoding='utf-8')
    text = text_file.read()

    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), 
                                         use_mmr=True, diversity=0.7, 
                                         seed_keywords=seed_keywords, top_n=5)
    return keywords

def generate_tags():
    tags = []
    manual_tags = collect_manual_tags()
    text_files = next(walk("webpageTextFiltered/"), (None, None, []))[2]
    for names in text_files:
        print(names)
        for keyword_pair in generate_tags_one_webpage("webpageTextFiltered/" + names, manual_tags):
            if keyword_pair[1] > 0.5 and keyword_pair[0] not in tags:
                tags.append(keyword_pair[0])
    return tags

def embed_tags(tags):
    roberta_path = "modules/roberta-large"

    tokenizer = RobertaTokenizer.from_pretrained(roberta_path, local_files_only=True)
    encoder = RobertaModel.from_pretrained(roberta_path, local_files_only=True)
    tokens = tokenizer(tags, return_tensors='pt', padding=True)
    embeddings = encoder(input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"])["last_hidden_state"]
    return embeddings[:, 0].detach().numpy()

# ----- Generate tags based on manual tags and output to a file for further use -----
# generated_tags = generate_tags()
# with open("generated_tags.txt", 'w', encoding='utf-8') as out_file:
#     for tag in generated_tags:
#         out_file.write(tag + "\n")
# -----------------------------------------------------------------------------------

# ----- Use spectral clustering to cluster all tags ---------------------------------
# generated_tags = []
# with open("generated_tags.txt") as tags_file:
#     generated_tags = tags_file.read().splitlines()
# generated_tags = generated_tags + collect_manual_tags()
# tags_embeddings = embed_tags(generated_tags)
# print(tags_embeddings.shape)
# num_clusters = 60
# clustering = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0).fit(tags_embeddings)

# clustered_tags = []
# for j in range(num_clusters):
#     clustered_tags.append([])
# for i in range(len(clustering.labels_)):
#     clustered_tags[clustering.labels_[i]].append(generated_tags[i])
# with open("clustered_tags.txt", 'w', encoding='utf-8') as out_file:
#     for i in range(num_clusters):
#         for tag in clustered_tags[i]:
#             out_file.write(tag + "\n")
#         out_file.write("\n")
# -----------------------------------------------------------------------------------

# ----- Use HDBSCAN to cluster all tags ----------------------------------------------
# generated_tags = []
# with open("generated_tags.txt") as tags_file:
#     generated_tags = tags_file.read().splitlines()
# generated_tags = generated_tags + collect_manual_tags()
# tags_embeddings = embed_tags(generated_tags)

# # nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(tags_embeddings)
# # distances, indices = nbrs.kneighbors(tags_embeddings)
# # print(np.mean(distances, axis = 1))
# # print(np.mean(distances, axis = 1).shape)
# # print(np.mean(distances))

# clustering = HDBSCAN(min_cluster_size=2, max_cluster_size=20).fit(tags_embeddings)
# num_clusters = max(clustering.labels_) + 1
# print("No of original tags: ", len(generated_tags))
# print("No of clusters: ", num_clusters)

# clustered_tags = []
# for j in range(num_clusters):
#     clustered_tags.append([])
# for i in range(len(clustering.labels_)):
#     if clustering.labels_[i] != -1:
#         clustered_tags[clustering.labels_[i]].append(generated_tags[i])
# with open("clustered_tags.txt", 'w', encoding='utf-8') as out_file:
#     for i in range(num_clusters):
#         for tag in clustered_tags[i]:
#             out_file.write(tag + "\n")
#         out_file.write("\n")
# -----------------------------------------------------------------------------------