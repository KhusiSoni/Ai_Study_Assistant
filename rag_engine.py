import os
import re
import uuid
import hashlib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 800
TOP_K_RETRIEVE = 8
TOP_K_FINAL = 5


def extract_text_from_pdf(path):
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing")
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")


def extract_text_from_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text.strip()
    except ImportError:
        raise RuntimeError("python-docx is required for DOCX parsing")
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX: {e}")


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def semantic_chunk_text(text, min_chars=MIN_CHUNK_CHARS, max_chars=MAX_CHUNK_CHARS):
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    buffer = []
    buf_len = 0

    for para in paragraphs:
        para_len = len(para)
        if buf_len + para_len <= max_chars:
            buffer.append(para)
            buf_len += para_len
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
            buffer = [para]
            buf_len = para_len

    if buffer:
        chunks.append("\n\n".join(buffer))

    if len(chunks) == 1 and len(chunks[0]) < min_chars:
        return chunks

    merged = []
    for c in chunks:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)

    return merged if merged else [text]


class RAGEngine:
    def __init__(self, storage_dir="uploads"):
        self.storage_dir = storage_dir
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            sublinear_tf=True,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
        )
        self.documents = {}
        self.chunks = []
        self.chunk_meta = []
        self.chunk_vectors = None
        self._fitted = False
        os.makedirs(storage_dir, exist_ok=True)

    def load_document(self, filepath, doc_id=None, display_name=None):
        ext = os.path.splitext(filepath)[1].lower()
        parsers = {".pdf": extract_text_from_pdf, ".txt": extract_text_from_txt, ".docx": extract_text_from_docx}
        parser = parsers.get(ext)
        if not parser:
            raise ValueError(f"Unsupported file type: {ext}")

        text = parser(filepath)
        if not text:
            raise ValueError("No text could be extracted from the file.")

        doc_id = doc_id or str(uuid.uuid4())[:8]
        if display_name is None:
            display_name = os.path.basename(filepath)

        chunks = semantic_chunk_text(text)
        self.documents[doc_id] = {
            "text": text,
            "chunks": chunks,
            "filepath": filepath,
            "display_name": display_name,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        for i, _chunk in enumerate(chunks):
            self.chunk_meta.append({"doc_id": doc_id, "chunk_idx": start_idx + i, "local_idx": i})
        self._rebuild_index()
        return doc_id

    def remove_document(self, doc_id):
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        new_chunks = []
        new_meta = []
        for c, m in zip(self.chunks, self.chunk_meta):
            if m["doc_id"] != doc_id:
                new_chunks.append(c)
                new_meta.append(m)
        self.chunks = new_chunks
        self.chunk_meta = new_meta
        self._rebuild_index()
        return True

    def _rebuild_index(self):
        if not self.chunks:
            self._fitted = False
            self.chunk_vectors = None
            return
        try:
            self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
            self._fitted = True
        except Exception:
            self._fitted = False
            self.chunk_vectors = None

    def retrieve(self, query, top_k=TOP_K_FINAL):
        if not self._fitted or self.chunk_vectors is None or self.chunk_vectors.shape[0] == 0:
            return []
        try:
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.chunk_vectors).flatten()

            query_terms = set(query.lower().split())
            term_boost = np.zeros(len(self.chunks))
            for i, chunk in enumerate(self.chunks):
                chunk_lower = chunk.lower()
                overlap = sum(1 for t in query_terms if t in chunk_lower)
                term_boost[i] = overlap / max(len(query_terms), 1)

            combined = scores * 0.7 + term_boost * 0.3
            top_indices = np.argsort(combined)[::-1][:top_k]

            results = []
            seen_chunks = set()
            for idx in top_indices:
                if combined[idx] < 0.02:
                    continue
                if idx in seen_chunks:
                    continue
                seen_chunks.add(idx)
                meta = self.chunk_meta[idx]
                doc_id = meta["doc_id"]
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "combined_score": float(combined[idx]),
                    "doc_id": doc_id,
                    "chunk_idx": meta["chunk_idx"],
                    "local_idx": meta["local_idx"],
                    "document": self.get_document_info(doc_id),
                })
            return results
        except Exception:
            return []

    def get_document_info(self, doc_id):
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        return {
            "doc_id": doc_id,
            "filepath": os.path.basename(doc["filepath"]),
            "display_name": doc["display_name"],
            "chunks": len(doc["chunks"]),
            "char_count": doc["char_count"],
            "word_count": doc["word_count"],
        }

    def list_documents(self):
        docs = []
        for did in self.documents:
            info = self.get_document_info(did)
            if info:
                docs.append(info)
        return docs

    def clear(self):
        self.documents = {}
        self.chunks = []
        self.chunk_meta = []
        self.chunk_vectors = None
        self._fitted = False
