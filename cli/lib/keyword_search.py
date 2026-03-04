import math
import pickle
import os
import string
from lib.search_utils import load_movies, load_stopwords, CACHE_PATH , BM25_K1, BM25_B
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) 
        self.docmap = {} # map document ID to document
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = CACHE_PATH/'doc_lengths.pkl'
         

        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}" 
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")

        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count+1) / (term_doc_count + 1)) 
    
    def get_bm25_tf(self, doc_id ,term, k1 = BM25_K1, b= BM25_B ):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b*(doc_length/avg_doc_length)
        else:
            length_norm = 1

        return (tf * (k1 + 1)) / (tf + k1*length_norm)


    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5)/(term_doc_count + 0.5) + 1)

    def bm25_search(self, query, limit = 5):
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0
            for token in query_tokens:
                score += self.get_bm25_tfidf(doc_id, token)
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True )

        results = sorted_scores[:limit]

        formatted_results = []
        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            formatted_results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "score": score,
                    'description': self.docmap[doc_id]['description']
                }
            )
        return formatted_results

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf*idf
    
    def get_bm25_tfidf(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf*bm25_idf
    
    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        if len(lengths) == 0:
            return 0.0
        total = 0
        for l in lengths:
            total += l
        return total / len(lengths)

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open( self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies,f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths,f)

    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load( f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load( f)
        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)
         
def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"first document for token 'merida' = {docs[0]} ")

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    # CLI passes doc_id as string; index uses int IDs from data.
    tf_idf = idx.get_tfidf(int(doc_id), term)
    print(f"TF-IDF score of '{term}' in document  '{doc_id}': {tf_idf:.2f}")

def bm25_tf_command(doc_id, term, k1 = BM25_K1, b = BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(int(doc_id), term, k1, b)

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_search(query, limit = 5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def clean_text(text):
    text=  text.lower()
    text = text.translate(str.maketrans("","",string.punctuation)) 
    return text


def tokenize_text(text):
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        tok = tok.strip('\n')
        if tok and tok not in  stopwords:
            return True
        return False

    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res

def has_matching_token(query_toks, movie_toks):
    for query_tok in query_toks:
        for movie_tok in movie_toks:
            if query_tok in movie_tok:
                return True
    return False

def search_command(query, n_results = 5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        matching_docs_ids = idx.get_documents(qt)
        for matchin_doc_id in matching_docs_ids:
            if matchin_doc_id in seen:
                continue
            seen.add(matchin_doc_id)
            matching_doc = idx.docmap[matchin_doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res

        
    for movie in movies:
        movie_tokens = tokenize_text(movie['title'])
        if has_matching_token(query_tokens, movie_tokens):
            res.append(movie)
        if len(res) == n_results:
            break
    return res

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    # CLI passes doc_id as string; index uses int IDs from data.
    print(idx.get_tf(int(doc_id), term))
