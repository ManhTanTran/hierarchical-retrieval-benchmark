from dapr_hhr.retrieval.hashing import HashingRetriever


def test_hashing_retriever_is_deterministic_and_filters_candidates():
    items = {"a": "capital paris france", "b": "mars phobos", "c": "ocean"}
    first = HashingRetriever(64).fit(items)
    second = HashingRetriever(64).fit(items)
    first_results = first.search("capital france", 3, candidate_ids=["a", "c"])
    second_results = second.search("capital france", 3, candidate_ids=["a", "c"])
    assert first_results == second_results
    assert {result.item_id for result in first_results} == {"a", "c"}
