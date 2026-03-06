from nltk.metrics.scores import precision
import json

from lib.search_utils import DATA_PATH, load_movies
from lib.hybrid_search import HybridSearch


def load_test_cases():
    with open(DATA_PATH/'golden_dataset.json') as f:
        test_cases = json.load(f)['test_cases']
    return test_cases

def evaluate(limit):
    test_cases = load_test_cases()
    movies = load_movies()

    hs = HybridSearch(movies)

    for test_case in test_cases:
        qry = test_case['query']
        exp = test_case['relevant_docs']

        rrf_results = hs.rrf_search(query=qry, limit=limit, k=60)
        relevant_cnt = 0
        for rrf_result in rrf_results:
            relevant_cnt += rrf_result['title'] in exp
        
        precision = relevant_cnt / limit
        recall = relevant_cnt / len(exp)
        
        print(qry)
        retrieved = ", ".join([r['title'] for r in rrf_results]) 

        print(f"- Precision@{limit}: {precision:.4f}")
        print(f"- Recall@{limit}: {recall:.4f}")
        print(f"- Retrieved: {retrieved}")
        print(f"- Relevant: {", ".join(exp)}")


