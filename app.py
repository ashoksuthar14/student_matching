import os
import math
from typing import List, Tuple, Dict, Any

from flask import Flask, render_template, jsonify

# Supabase (Python client v2)
from supabase import create_client, Client

# Gemini (Google Generative AI) for embeddings
import google.generativeai as genai
from dotenv import load_dotenv


# Hardcoded Supabase credentials as requested
HARDCODED_SUPABASE_URL = "https://daiuefevbtkthizqykez.supabase.co"
HARDCODED_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRhaXVlZmV2YnRrdGhpenF5a2V6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1Njg1ODIsImV4cCI6MjA3NDE0NDU4Mn0.S8K7oRe2aTAa9ci0BlBZ26GOAn0O7WCFFwdl5o8gi-Y"
)


def create_app() -> Flask:
    app = Flask(__name__)

    # Load environment variables from .env if present
    load_dotenv()

    # Initialize Supabase client
    app.config["SUPABASE_URL"] = HARDCODED_SUPABASE_URL
    app.config["SUPABASE_KEY"] = HARDCODED_SUPABASE_KEY
    app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "AIzaSyAsutlZP3xvpyGEMiRmWa-8GKhLaKjxb34")

    app.supabase: Client = create_client(
        app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"]
    )

    # Configure Gemini
    if app.config["GEMINI_API_KEY"]:
        genai.configure(api_key=app.config["GEMINI_API_KEY"])

    @app.route("/")
    def index():
        return jsonify({
            "status": "ok",
            "routes": ["/student_matching"],
        })

    @app.route("/student_matching")
    def student_matching():
        try:
            students = fetch_students(app.supabase)
        except Exception as exc:  # pragma: no cover - surface error to UI
            return render_template(
                "student_matching.html",
                error=f"Failed to fetch students: {exc}",
                students=[],
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        if not students:
            return render_template(
                "student_matching.html",
                error="No students found.",
                students=[],
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        # If no Gemini key present, show message
        if not app.config["GEMINI_API_KEY"]:
            return render_template(
                "student_matching.html",
                error=(
                    "GEMINI_API_KEY not set. Create a .env with GEMINI_API_KEY and restart."
                ),
                students=students,
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        try:
            match_pair, similarity = match_best_pair_with_gemini(students)
            match_reason = generate_match_reason(match_pair[0], match_pair[1])
        except Exception as exc:  # pragma: no cover
            return render_template(
                "student_matching.html",
                error=f"Failed to compute matches: {exc}",
                students=students,
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        return render_template(
            "student_matching.html",
            error=None,
            students=students,
            match_pair=match_pair,
            similarity=similarity,
            match_reason=match_reason,
        )

    return app


def fetch_students(supabase: Client) -> List[Dict[str, Any]]:
    """Fetch students with summaries from Supabase.

    Expected table: chat_summaries(id, student_name, summary, diagnoses, created_at)
    """
    # Adjust table/columns here if your schema differs
    response = supabase.table("chat_summaries").select(
        "student_name, summary"
    ).execute()
    data = response.data or []

    # Normalize to required fields
    normalized: List[Dict[str, Any]] = []
    for row in data:
        if row.get("summary"):
            normalized.append(
                {
                    "name": row.get("student_name") or "Student",
                    "summary": row.get("summary"),
                }
            )
    return normalized


def embed_texts_with_gemini(texts: List[str]) -> List[List[float]]:
    """Return embeddings for list of texts using Gemini embedding model."""
    # Use the recommended text embedding model
    embeddings: List[List[float]] = []
    for text in texts:
        # Use top-level embed_content helper as recommended
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        # Normalize various possible response shapes
        values: List[float] = []
        if isinstance(result, dict):
            if "embedding" in result:
                emb = result["embedding"]
                if isinstance(emb, dict):
                    values = emb.get("values") or emb.get("value") or []
                elif isinstance(emb, list):
                    values = emb
            elif "data" in result:
                # Some SDKs wrap embeddings in result['data'][0]['embedding']
                data = result.get("data") or []
                if isinstance(data, list) and data:
                    first = data[0] or {}
                    emb = first.get("embedding")
                    if isinstance(emb, dict):
                        values = emb.get("values") or emb.get("value") or []
                    elif isinstance(emb, list):
                        values = emb
        else:
            # Fallback for object-style response
            emb_obj = getattr(result, "embedding", None)
            if isinstance(emb_obj, list):
                values = emb_obj
            else:
                values = getattr(emb_obj, "values", None) or getattr(emb_obj, "value", None) or []
        embeddings.append(list(values))
    return embeddings


def summarize_texts_with_gemini(texts: List[str]) -> List[str]:
    """Summarize each input text using Gemini 2.5 Flash to standardize content before embedding."""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    summaries: List[str] = []
    for text in texts:
        prompt = (
            "Summarize the following student's description in 2-3 concise sentences, "
            "focusing on key traits, interests, and support needs relevant for pairing.\n\n"
            f"Text:\n{text}"
        )
        try:
            resp = model.generate_content(prompt)
            summarized = getattr(resp, "text", None) or ""
        except Exception:
            summarized = text  # Fallback to original if generation fails
        summaries.append(summarized.strip() or text)
    return summaries


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def match_best_pair_with_gemini(
    students: List[Dict[str, Any]]
) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], float]:
    """Find the most similar pair of students based on summary embeddings."""
    if len(students) < 2:
        raise ValueError("Need at least two students to match")

    texts = [s["summary"] for s in students]
    # First, normalize/summarize using Gemini 2.5 Flash, then embed the summaries
    summarized_texts = summarize_texts_with_gemini(texts)
    vectors = embed_texts_with_gemini(summarized_texts)

    best_i, best_j, best_score = 0, 1, -1.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = cosine_similarity(vectors[i], vectors[j])
            if score > best_score:
                best_score = score
                best_i, best_j = i, j

    return (students[best_i], students[best_j]), best_score


def generate_match_reason(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Generate a concise, friendly reason why two students match well."""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    prompt = (
        "Given two student summaries, explain in 1-2 short sentences why they make a good match. "
        "Be specific but concise, referencing complementary interests, goals, styles, or support needs.\n\n"
        f"Student A (name: {a.get('name','Student A')}):\n{a.get('summary','')}\n\n"
        f"Student B (name: {b.get('name','Student B')}):\n{b.get('summary','')}\n\n"
        "Response:"
    )
    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "")
        return (text or "They share compatible interests and goals, making them a strong match.").strip()
    except Exception:
        return "They share compatible interests and goals, making them a strong match."


"""Expose a module-level Flask app for WSGI servers (e.g., gunicorn)."""
app = create_app()

if __name__ == "__main__":
    # For local development
    app.run()

