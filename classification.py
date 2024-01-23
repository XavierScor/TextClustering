from solution_similarity_score import similariy_score_one_webpage
from os import walk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from keyword_extraction import collect_manual_descriptions, collect_titles, collect_manual_tags

tags = ["air quality", # Environment protection related
        "ghg, co2, carbon emission",
        # "eco-friendly",
        "recycle",
        "protection",
        # "sustainable",
        "affordable and clean energy", # Energy related
        "energy efficiency",
        "sustainable energy",
        "renewable energy",
        "cool roofs", # Architectual solutions
        "paving",
        "smart cities and smart growth and smart grids",
        # "architectual",
        "awarness and preparedness", # Abstract solution without concrete implementation
        "sensitization",
        "rating system",
        "training",
        "policy",
        "officer position",
        "health policy",
        "health risks",
        "promotion",
        "interdisciplinary cooperation",
        "public transport",
        # "climate change and issues", # Climate change and disaster
        "disaster",
        # "heat island",
        "innovation and infrastructure", # Innovation related
        "innovating material",
        "salt water", # Environment context requirement
        "stormwater and rainwater",
        "airflow",
        "drainage system",
        "elastocaloric", # Cooling solutions
        "air-conditioning",
        "biomimicry",
        "absorption",
        "reflection",
        "radiation",
        "passive",
        "evaporative and ventilation",
        # "heat resilience",
        "geothermal"]

generated_tags = []
with open("generated_tags.txt") as tags_file:
    generated_tags = tags_file.read().splitlines()
tags = generated_tags

scale_tags = ["single person", "building scale", "urban scale", "district scale"]
scale_result = []
text_files = next(walk("webpageTextFiltered/"), (None, None, []))[2]
classification_result = np.zeros((len(text_files), len(tags) + 1))
most_related_tag = []
file_processed = 0
for names in text_files:
    text_id = int(names[8:-4])
    classification_result[file_processed, 0] = text_id

    text_path = "webpageTextFiltered/" + names
    scores = similariy_score_one_webpage(text_path, tags)
    print(names + " " + scores[0][0] + ", " + str(scores[0][1]))

    for score_pair in scores:
        classification_result[file_processed, tags.index(score_pair[0]) + 1] = score_pair[1]

    most_related_tag.append(scores[0][0] + ", " + scores[1][0] + ", " + scores[2][0])
    
    scale_scores = similariy_score_one_webpage(text_path, scale_tags)
    scale_result.append(scale_tags.index(scale_scores[0][0]))

    file_processed = file_processed + 1

with open("classification_result.txt", 'w', encoding='utf-8') as out_file:
    for i in range(len(text_files)):
        out_file.write(str(int(classification_result[i][0])) + " ")
        for j in range(1, len(tags) + 1):
            out_file.write(str(classification_result[i][j]) + " ")
        out_file.write("\n")

classification_result_without_index = classification_result[:, 1:]
pca = PCA(n_components=2)
pca.fit(classification_result_without_index)
classification_result_without_index_projected = pca.transform(classification_result_without_index)
descriptions = collect_manual_descriptions()
titles = collect_titles()

# hdb = HDBSCAN(min_cluster_size = 2, max_cluster_size = 20)
hdb = KMeans(n_clusters=10, random_state=0, n_init="auto")
hdb.fit(classification_result_without_index)
cluster_id = hdb.labels_
print("Number of clusters: ", max(cluster_id) + 2)

with open("classification_result_projected.csv", 'w', encoding='utf-8') as out_file:
    for i in range(len(text_files)):
        out_file.write("\"" + str(int(classification_result[i][0])) + "\",")
        out_file.write("\"" + str(classification_result_without_index_projected[i][0]) + "\",")
        out_file.write("\"" + str(classification_result_without_index_projected[i][1]) + "\",")
        out_file.write("\"" + str(cluster_id[i]) + "\",")
        out_file.write("\"" + str(scale_result[i]) + "\",")
        out_file.write("\"" + titles[int(classification_result[i][0])] + ": " + str(most_related_tag[i]) + "\",")
        # out_file.write("\"" + descriptions[int(classification_result[i][0])] + "\"")
        out_file.write("\n")
