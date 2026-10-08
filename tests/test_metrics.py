from src.medvision.metrics import labels_to_vector, extract_predicted_labels, rouge_l


def test_label_roundtrip():
    labels = ["Cardiomegaly", "Pleural Effusion"]
    vec = labels_to_vector(labels)
    assert sum(vec) == 2


def test_extract_predicted_labels():
    text = "Possible cardiomegaly and pleural effusion."
    found = extract_predicted_labels(text)
    assert "Cardiomegaly" in found
    assert "Pleural Effusion" in found


def test_rouge_identity():
    assert rouge_l("no acute finding", "no acute finding") == 1.0
