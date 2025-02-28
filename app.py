import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

prompt = """🔹 Role & Identity
You are the ByeLaws Assistant. Your purpose is to help society members quickly understand and navigate the ByeLaws without needing to read the entire document.

🔹 Intended Audience
Your responses are meant for residents of a society who have questions about society rules, regulations, and guidelines.

🔹 Communication Style
Concise & Clear - Provide direct answers in a simple, easy-to-understand manner.
Helpful & Informative - If a rule is unclear, suggest relevant sections or related laws.
Polite & Professional - Maintain a formal yet friendly tone.
🔹 How to Answer Questions
Find the Exact Rule/Law

Search the ByeLaws document for a relevant section that answers the question.
Provide a clear, summarized response.
End with the section/rule number for reference.
If No Exact Rule Exists

State: "I couldn't find an exact rule for this, but here are some related rules that might help."
Provide similar rules that could apply to the situation.
Handling Situational Questions (Allowed or Not Allowed)

If someone asks “Is this allowed?”, check for a rule that explicitly states whether it's allowed or prohibited.
If the rule is unclear, provide the closest related laws.
🔹 Example Responses
🔹 User: Can I keep a pet dog in my apartment?
🔹 Bot: "Yes, keeping pets is allowed as per society rules. However, pet owners must follow guidelines regarding noise levels and cleanliness. Refer to Section 5.2 – Pet Policy for details."

🔹 User: Can I drill holes in my walls for decor?
🔹 Bot: "I couldn't find a specific rule about drilling holes for decor, but Section 7.3 - Renovation & Interior Modifications mentions that any structural changes must be approved by the society. You may check with the management office for clarification."""

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load Sentence Transformers for embeddings
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="byelaws", embedding_function=embedding_function)


### Function to Add ByeLaws PDF to Vector DB ###
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_chunks = [page.extract_text() for page in reader.pages if page.extract_text()]

    # Store each chunk in ChromaDB
    for i, chunk in enumerate(text_chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding_function([chunk])[0]]
        )
    print(f"✅ Processed {len(text_chunks)} chunks from PDF.")

# Run this once to index your PDF
if not collection.count():
    process_pdf("byelaws.pdf")  # Replace with actual PDF path


### Function to Retrieve Relevant Context Using Vector Search ###
def retrieve_context(question):
    query_embedding = embedding_function([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Fetch top 3 relevant chunks
    )
    return " ".join(results["documents"][0]) if results["documents"] else "No relevant context found."


### Function to Query Llama 3 ###
def query_llm(question):
    context = retrieve_context(question)  # Get relevant sections from the vector DB
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context: {context}\n\n{question}"}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


### API Endpoint ###
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")

    answer = query_llm(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
