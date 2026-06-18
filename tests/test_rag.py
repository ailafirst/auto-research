"""RAG 服务测试。"""

from __future__ import annotations

from app.models.source import CrawledDocument, EvidenceChunk
from app.services.rag_service import TextChunker


def _make_sample_doc() -> CrawledDocument:
    return CrawledDocument(
        url="https://example.com/article",
        title="测试文章",
        content=(
            "人工智能在2024年取得了重大突破。\n\n"
            "深度学习模型在医疗诊断中表现优异，准确率超过90%。\n\n"
            "专家预计AI将在未来五年改变多个行业。\n\n"
            "自然语言处理技术也取得了显著进步。\n\n"
            "多模态AI模型成为新的研究热点。\n\n"
            "AI Agent 技术在自动化领域展现出巨大潜力。"
        ),
    )


def test_text_chunker_basic() -> None:
    """测试基本切片功能。"""
    chunker = TextChunker(chunk_size=200, overlap=20)
    doc = _make_sample_doc()
    chunks = chunker.chunk_document(doc, task_id="test_001", source_id="src_001")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.task_id == "test_001"
        assert chunk.source_id == "src_001"
        assert chunk.text
        assert chunk.url == doc.url


def test_text_chunker_small_doc() -> None:
    """测试小文档切片。"""
    chunker = TextChunker(chunk_size=5000, overlap=100)
    doc = CrawledDocument(
        url="https://example.com/short",
        title="短文章",
        content="这是一篇很短的文章。只有几句话。"
    )
    chunks = chunker.chunk_document(doc, task_id="test_002", source_id="src_002")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_text_chunker_empty_doc() -> None:
    """测试空文档切片。"""
    chunker = TextChunker()
    doc = CrawledDocument(
        url="https://example.com/empty",
        title="空文章",
        content=""
    )
    chunks = chunker.chunk_document(doc, task_id="test_003", source_id="src_003")
    assert len(chunks) == 0


def test_text_chunker_chunk_order() -> None:
    """测试 Chunk 顺序正确。"""
    chunker = TextChunker(chunk_size=50, overlap=10)
    doc = _make_sample_doc()
    chunks = chunker.chunk_document(doc, task_id="test_004", source_id="src_004")

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i

    assert chunks[0].chunk_index < chunks[-1].chunk_index


def test_evidence_chunk_model() -> None:
    """测试 EvidenceChunk 数据模型。"""
    chunk = EvidenceChunk(
        task_id="test_001",
        source_id="src_001",
        url="https://example.com",
        title="测试",
        chunk_index=0,
        text="测试文本内容",
    )
    assert chunk.chunk_id.startswith("chunk_")
    assert chunk.task_id == "test_001"
    assert chunk.chunk_index == 0
    assert chunk.created_at
