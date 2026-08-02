import pytest

from dapr_hhr.data import DatasetBundle, Document, Passage, Query, validate_dataset


def test_validation_rejects_missing_document_reference():
    bundle = DatasetBundle(
        documents=[Document("d1", "Title", "Body")],
        passages=[Passage("p1", "missing", "Title", "Body")],
        queries=[Query("q1", "query")],
        qrels=[],
    )
    with pytest.raises(ValueError, match="missing documents"):
        validate_dataset(bundle)

