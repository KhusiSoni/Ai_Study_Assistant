import os
import uuid
import json
import time
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine

app = Flask(__name__)
app.secret_key = os.urandom(32).hex()
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sessions = {}


def get_engine():
    sid = session.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())[:12]
        session["session_id"] = sid
    if sid not in sessions:
        sess_dir = os.path.join(UPLOAD_FOLDER, f"session_{sid}")
        os.makedirs(sess_dir, exist_ok=True)
        sessions[sid] = RAGEngine(storage_dir=sess_dir)
    return sessions[sid]


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PDF, DOCX, or TXT."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    try:
        engine = get_engine()
        doc_id = engine.load_document(filepath, display_name=file.filename)
        info = engine.get_document_info(doc_id)
        docs = engine.list_documents()
        return jsonify({"status": "ok", "doc_id": doc_id, "info": info, "documents": docs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    engine = get_engine()
    history = data.get("history", [])

    if not engine.list_documents():
        return jsonify({
            "answer": "Please upload a document first so I can look up the answer in your material.",
            "sources": [],
            "context": None,
            "empty": True,
        }), 200

    results = engine.retrieve(question, top_k=8)
    if not results:
        return jsonify({
            "answer": "I couldn't find relevant information in your uploaded materials. Try rephrasing your question or uploading a more specific document.",
            "sources": [],
            "context": None,
        }), 200

    answer = generate_answer(question, results, history)

    source_docs = list({r["doc_id"]: r["document"] for r in results}.values())
    context_data = [
        {"doc_id": r["doc_id"], "source": r["document"]["display_name"], "chunk": r["chunk"], "score": round(r["combined_score"], 3)}
        for r in results
    ]

    return jsonify({
        "answer": answer,
        "sources": source_docs,
        "chunks": context_data,
    }), 200


@app.route("/api/ask/stream", methods=["POST"])
def ask_stream():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    engine = get_engine()
    history = data.get("history", [])

    if not engine.list_documents():
        def empty_gen():
            yield json.dumps({"type": "error", "text": "Please upload a document first."}) + "\n"
        return Response(stream_with_context(empty_gen()), mimetype="application/x-ndjson")

    results = engine.retrieve(question, top_k=8)
    if not results:
        def no_results_gen():
            yield json.dumps({"type": "error", "text": "No relevant content found. Try rephrasing."}) + "\n"
        return Response(stream_with_context(no_results_gen()), mimetype="application/x-ndjson")

    source_docs = list({r["doc_id"]: r["document"] for r in results}.values())
    context_data = [
        {"doc_id": r["doc_id"], "source": r["document"]["display_name"], "chunk": r["chunk"], "score": round(r["combined_score"], 3)}
        for r in results
    ]

    def generate():
        yield json.dumps({"type": "meta", "sources": source_docs, "chunks": context_data}) + "\n"

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                for token in _stream_openai_answer(question, [r["chunk"] for r in results], history, openai_key):
                    yield json.dumps({"type": "token", "text": token}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return
            except Exception:
                pass

        answer = _local_answer(question, [r["chunk"] for r in results])
        yield json.dumps({"type": "token", "text": answer}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def generate_answer(question, results, history=None):
    chunks = [r["chunk"] for r in results]
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            return _openai_answer(question, chunks, history, openai_key)
        except Exception:
            pass
    return _local_answer(question, chunks)


def _build_openai_context(chunks, history=None):
    parts = []
    for i, c in enumerate(chunks[:6]):
        parts.append(f"[{i+1}] {c}")
    context = "\n\n".join(parts)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise study assistant. Answer questions ONLY using the provided excerpts. "
                "Rules:\n"
                "1. Cite sources inline using [1], [2], etc.\n"
                "2. If excerpts lack info, say 'This material does not cover that.'\n"
                "3. Keep answers concise but thorough.\n"
                "4. Use bullet points and formatting for readability.\n"
                "5. If the question is a follow-up, use prior conversation context."
            ),
        }
    ]
    if history:
        for h in history[-4:]:
            messages.append({"role": "user", "content": h.get("question", "")})
            messages.append({"role": "assistant", "content": h.get("answer", "")})
    messages.append({
        "role": "user",
        "content": f"EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER (with citations):",
    })
    return messages


def _openai_answer(question, chunks, history, api_key):
    import requests
    messages = _build_openai_context(chunks, history)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2, "max_tokens": 800},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _stream_openai_answer(question, chunks, history, api_key):
    import requests
    messages = _build_openai_context(chunks, history)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2, "max_tokens": 800, "stream": True},
        stream=True,
        timeout=30,
    )
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data = decoded[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
            except json.JSONDecodeError:
                continue


def _local_answer(question, chunks):
    if not chunks:
        return "No relevant content found."

    best = chunks[0]
    summary_words = best.split()[:80]
    summary = " ".join(summary_words)
    if len(best.split()) > 80:
        summary += "..."

    lines = best.strip().split("\n")
    key_points = [l.strip() for l in lines if l.strip() and len(l.strip()) > 30][:3]

    answer = f"**Relevant excerpt from your material:**\n\n{best}"
    if len(chunks) > 1:
        answer += f"\n\n---\n*Found {len(chunks)} relevant passages. Upload more materials or set `OPENAI_API_KEY` for AI-generated answers with citations.*"

    return answer


@app.route("/api/documents", methods=["GET"])
def list_docs():
    engine = get_engine()
    docs = engine.list_documents()
    return jsonify({"documents": docs}), 200


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    engine = get_engine()
    if engine.remove_document(doc_id):
        return jsonify({"status": "ok", "documents": engine.list_documents()}), 200
    return jsonify({"error": "Document not found"}), 404


@app.route("/api/clear", methods=["POST"])
def clear():
    sid = session.get("session_id")
    if sid and sid in sessions:
        sessions[sid].clear()
    session.pop("session_id", None)
    return jsonify({"status": "ok"}), 200


@app.route("/api/session", methods=["GET"])
def session_info():
    engine = get_engine()
    docs = engine.list_documents()
    chunk_count = len(engine.chunks)
    return jsonify({
        "session_id": session.get("session_id", ""),
        "documents": docs,
        "total_chunks": chunk_count,
        "has_documents": len(docs) > 0,
    }), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.0"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"AI Study Assistant v2 -> http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
