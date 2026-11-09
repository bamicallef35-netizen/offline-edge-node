from pathlib import Path

from edge_node import HashedEmbedder, Index, chunk_text, grounded_prompt, load_docs, netguard

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_chunking_is_deterministic_and_bounded() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(12))
    a = chunk_text(text, size=500, overlap=1)
    b = chunk_text(text, size=500, overlap=1)
    assert a == b
    assert len(a) > 1
    assert all(len(c) <= 500 + 250 for c in a)       # a single long paragraph may exceed size; windows do not
    # overlap carries the last paragraph of one window into the next
    assert a[0].split("\n\n")[-1] == a[1].split("\n\n")[0]


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("") == []


def test_load_docs_reads_the_bundled_corpus() -> None:
    docs = load_docs(DOCS)
    assert "walter.md" in docs and "pask.md" in docs
    assert all(docs[k].strip() for k in docs)


def test_known_question_retrieves_the_right_document() -> None:
    index = Index(HashedEmbedder()).build(load_docs(DOCS))
    hits = index.query("Who built the tortoises Elmer and Elsie?", k=3)
    assert hits and hits[0][1].doc == "walter.md"
    hits = index.query("What is the Law of Requisite Variety?", k=3)
    assert hits[0][1].doc == "ashby.md"


def test_prompt_cites_the_retrieved_chunks() -> None:
    index = Index(HashedEmbedder()).build(load_docs(DOCS))
    hits = index.query("Musicolour and boredom", k=2)
    prompt = grounded_prompt("What was Musicolour?", hits)
    for _, chunk in hits:
        assert f"[{chunk.ref}]" in prompt
    assert "Question: What was Musicolour?" in prompt


def test_retrieval_runs_entirely_inside_the_air_gap() -> None:
    with netguard():
        index = Index(HashedEmbedder()).build(load_docs(DOCS))
        assert index.query("stigmergy", k=1)


def test_embeddings_are_unit_length_and_deterministic() -> None:
    e = HashedEmbedder(dim=512)
    v = e.embed(["the quick brown fox", "the quick brown fox", "an unrelated sentence"])
    assert abs(float((v[0] * v[0]).sum()) - 1.0) < 1e-5
    assert (v[0] == v[1]).all()
    assert float(v[0] @ v[2]) < float(v[0] @ v[1])
