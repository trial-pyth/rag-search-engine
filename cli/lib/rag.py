from llm import answer_question
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def rag(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60,limit = 5)
    rag_results = answer_question(query, rrf_results)
    print("Seach Results")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query, rrf_results)
    print(rag_results)

def doc_summarization(query, limit):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60,limit = limit)
    print("Seach Results")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query, rrf_results)
    print(rag_results)

        