from flask import Flask, render_template, request, jsonify
from rag_backend import load_and_index_documents, generate_rag_response

app = Flask(__name__)

# Load retriever once at startup
retriever_instance = load_and_index_documents()
if retriever_instance is None:
    raise RuntimeError("No documents indexed. Add PDFs to docs folder.")

chat_history = [
    {"bot": "Hi, I'm your AI assistant. Ask me anything about the company documentation."}
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", chat_history=chat_history)


@app.route("/ask", methods=["POST"])
def ask():
    global chat_history

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"user": "", "bot": "Please enter a question."})

    response = generate_rag_response(query, retriever_instance)

    chat_history.append({
        "user": query,
        "bot": response
    })


    return jsonify({"user": query, "bot": response})


if __name__ == "__main__":
    app.run(debug=True)
