# 🎓 Intelligent College Assistant

An AI-powered **College Assistant** built using **LangGraph**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Google Gemini**, and **Streamlit**, and evaluated with **RAGAS**.

The assistant intelligently classifies student queries into different categories and routes them to the appropriate knowledge source before generating an accurate, context-aware response.

---

## 🚀 Features

- 🤖 AI-powered conversational assistant
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 LLM-based query classification
- 🔀 Conditional routing using LangGraph
- 📘 Academic Handbook retrieval
- 💳 Fee Structure retrieval
- 💬 General conversation support
- 🎓 Personalized responses based on the student's programme
- 🌐 Interactive Streamlit interface
- ⚡ Fast semantic search using FAISS
- 📊 RAG evaluation using RAGAS

---

## 🏗️ System Architecture

```text
                     Student Question
                            │
                            ▼
                  LLM Query Classifier
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
 Academic Retriever    Fee Retriever      General Chat
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                 Response Generation
                            │
                            ▼
                 Personalized Response
```

---

## 🧠 Workflow

1. Student selects their programme.
2. Student enters a question.
3. The LLM classifies the query into one of three categories:
   - Academic
   - Fee
   - General
4. Depending on the category:
   - **Academic** → Retrieves relevant information from the Academic Handbook.
   - **Fee** → Retrieves relevant information from the Fee Structure.
   - **General** → Uses the LLM directly without document retrieval.
5. The assistant generates a personalized response based on the student's selected programme.

---

## 📚 Knowledge Sources

The assistant currently retrieves information from two official college documents:

- 📘 Academic Handbook
- 💳 Fee Structure

Both documents are:

- Loaded using **PyPDFLoader**
- Split using **Recursive Character Text Splitter** (chunk size 800, overlap 100)
- Embedded using **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- Indexed using **FAISS** (top 4 chunks retrieved per question)

---

## ⚙️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Interface |
| LangGraph | Workflow Orchestration |
| LangChain | LLM Framework |
| FAISS | Vector Store |
| HuggingFace Embeddings | Semantic Embeddings |
| Google Gemini (`gemini-3.5-flash-lite`) | Large Language Model |
| PyPDFLoader | PDF Processing |
| Recursive Character Text Splitter | Document Chunking |
| RAGAS | RAG Evaluation |
| python-dotenv | Environment Variables |

---

## 📁 Project Structure

```text
Intelligent_College_Assistant/
│
├── app.py                     # Streamlit user interface
├── main.py                    # Core LangGraph workflow (also runs as a terminal chat)
├── evaluate.py                # RAGAS evaluation script (run locally)
├── ragas_results_new.csv      # Detailed evaluation results, one row per question
├── academics_handbook.pdf     # Academic knowledge base
├── fee_structure.pdf          # Fee knowledge base
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚡ Installation

### Clone the repository

```bash
git clone https://github.com/AhmedMemon-007/langgraph-college-assistant.git
```

### Navigate to the project

```bash
git clone https://github.com/AhmedMemon-007/Project-intelligent-college-assistant.git
cd Project-intelligent-college-assistant
```

### Create and activate a virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> `langchain-community` is pinned to **0.4.1**, because RAGAS does not load with version 0.4.2.

### Create a `.env` file

```env
GOOGLE_API_KEY=your_api_key_here
```

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

### Run the application

```bash
streamlit run app.py
```

Or chat in the terminal:

```bash
python main.py
```

---

## 💬 Example Questions

### Academic

- What is the minimum attendance requirement?
- What are the graduation requirements?
- What is the grading policy?
- Can I appear in supplementary exams?

### Fee

- What is the tuition fee?
- Are scholarships available?
- What is the refund policy?
- What are the late payment charges?

### General

- Hello
- Who are you?
- Thank you
- What can you do?

---

## 🔀 Query Routing

| Query Type | Knowledge Source |
|------------|------------------|
| Attendance | Academic Handbook |
| Examinations | Academic Handbook |
| Credits | Academic Handbook |
| Degree Requirements | Academic Handbook |
| Tuition | Fee Structure |
| Scholarships | Fee Structure |
| Refund Policy | Fee Structure |
| Greetings | General LLM |

---

## 📊 Evaluation

The assistant was evaluated with **RAGAS** on **12 test questions** written from the two PDFs (6 academic, 6 fee). Each question has a reference answer taken directly from the documents. Gemini was used as the judge LLM.

### Results

| Metric | What it measures | Score |
|---|---|---|
| Context Recall | Did the retriever find all the information needed for the answer? | **1.00** |
| Context Precision | Were the retrieved chunks relevant, and ranked first? | **0.84** |
| Answer Relevancy | Does the answer respond to the question asked? | **0.78** |
| Faithfulness | Is every claim in the answer supported by the retrieved chunks? | **0.76** |
| Routing Accuracy | Was each question sent to the correct document? | **12 / 12** |

Detailed per-question scores are in [`ragas_results_new.csv`](ragas_results_new.csv).

### Findings

- **Retrieval is strong.** The correct information was retrieved for every question (recall 1.00), and the classifier routed all questions correctly.
- **All answers contained the correct facts.** Faithfulness and relevancy are lower because the friendly response prompt adds greetings, encouragement and extra details that are not in the documents. The judge counts these sentences as unsupported.
- **Context precision is lower for fee questions.** Each programme's fee table is split across several chunks, so for some questions the most relevant chunk is retrieved but not ranked first.

### Run the evaluation

```bash
python evaluate.py
```

The evaluation runs **locally only**. It is not part of the deployed app.

> **Note:** one run of 12 questions makes about 150 Gemini requests and takes about 30 minutes, because of the rate limiter. The Gemini free tier allows 500 requests per day per model.

---

## ⚠️ Limitations

- **Small test set:** 12 questions give a useful first signal, but a larger set would give more reliable scores.
- **Self-evaluation:** the same Gemini model generates and judges the answers, which may make scores slightly optimistic.
- **Table chunking:** fee tables are split by character count, so one table can be spread across chunks.
- **No source citations:** answers do not show which document section they came from.
- **API limits:** the free Gemini tier limits how many questions can be asked and evaluated per day.

---

## 🌟 Key Highlights

- Multi-source Retrieval-Augmented Generation (RAG)
- Intelligent LLM-based query routing
- State-based workflow using LangGraph
- Personalized responses
- Measured with RAGAS evaluation metrics
- Modular architecture
- Easily extensible to additional document sources

---

## 🚀 Future Improvements

- Stricter response prompt to raise faithfulness
- Table-aware chunking, or adding the programme name to each chunk, to raise context precision
- Larger evaluation set, with a different model as the judge
- Conversation memory
- Multi-agent workflow
- Hybrid Search (BM25 + FAISS)
- Citation of retrieved document sources
- Student login system
- Voice assistant
- Support for multiple universities
- Admin dashboard
- Course recommendation system
- Timetable assistant

---

## 👨‍💻 Author

**Muhammad Ahmed Memon**

- **GitHub:** https://github.com/AhmedMemon-007
- **LinkedIn:** https://www.linkedin.com/in/immohammadahmed/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
