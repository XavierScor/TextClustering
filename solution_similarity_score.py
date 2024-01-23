from keybert import KeyBERT, _mmr

def similariy_score_one_webpage(webpage_text_filename, candidates):
    text_file = open(webpage_text_filename, 'r', encoding='utf-8')
    text = text_file.read()
    
    kw_model = KeyBERT()
    doc_embeddings, word_embeddings = kw_model.extract_embeddings(text, candidates=candidates)
    scores = _mmr.mmr(doc_embeddings, word_embeddings, words=candidates, top_n=len(candidates))
    return scores

# candidates = ["individual scale", "building scale", "urban scale", "national scale", "global scale"]
# text_filename = "webpageTextFiltered/webpage_197.txt"
# print(similariy_score_one_webpage(text_filename, candidates))