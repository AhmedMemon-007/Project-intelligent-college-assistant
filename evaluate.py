from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,                      # Is the answer supported by the chunks? (no made-up facts)
    ResponseRelevancy,                 # Does the answer respond to the question?
    LLMContextPrecisionWithReference,  # Are the retrieved chunks useful?
    LLMContextRecall,                  # Did the retriever find ALL the needed information?
)

# The same graph, LLM and embeddings the assistant uses
from main import app, llm, embeddings


# 1. TEST QUESTIONS
# "reference" = the correct answer, taken from the PDFs.
# "route"     = the PDF the question should go to.

tests = [
    # Academic handbook 
    
    {
        "programme": "BBA",
        "route": "academic",
        "question": "How many grace marks can be given?",
        "reference": "A maximum of six grace marks can be given, in one paper or spread across several papers, "
                     "if the total marks are below 50 percent. The University gives them when results are finalised.",
    },
    {
        "programme": "BCA",
        "route": "academic",
        "question": "What do I need to be promoted from second year to third year?",
        "reference": "A student needs at least 90 percent of the credits of the first year examination and at least "
                     "50 percent of the credits of the second year examination.",
    },
    {
        "programme": "BCA",
        "route": "academic",
        "question": "How long is the summer training for BCA students?",
        "reference": "BCA students do four weeks of summer training after the fourth semester, "
                     "with a viva-voce in the fifth semester.",
    },
    {
        "programme": "BBA",
        "route": "academic",
        "question": "How long is the summer training for BBA students?",
        "reference": "BBA students do summer training for six to eight weeks after the fourth semester.",
    },
    {
        "programme": "B.Com (H)",
        "route": "academic",
        "question": "How many credits are needed to get the degree?",
        "reference": "A student needs 150 credits to be awarded the degree.",
    },
    {
        "programme": "BCA",
        "route": "academic",
        "question": "What is the maximum time allowed to complete the programme?",
        "reference": "The maximum time to complete any programme is five academic years, which is ten semesters.",
    },

    # Fee structure 
    {
        "programme": "BCA",
        "route": "fee",
        "question": "What is the total first year fee for my programme?",
        "reference": "The total first year fee for BCA is Rs. 1,20,000, including the Rs. 5,000 refundable security deposit.",
    },
    {
        "programme": "BBA",
        "route": "fee",
        "question": "What is the tuition fee per year for my programme?",
        "reference": "The academic (tuition) fee for BBA is Rs. 78,000 per year.",
    },
    {
        "programme": "B.Com (H)",
        "route": "fee",
        "question": "What is the total cost of my programme over three years?",
        "reference": "The total B.Com (H) programme cost over three years is Rs. 3,00,000, including the security deposit.",
    },
    {
        "programme": "BCA",
        "route": "fee",
        "question": "How do I pay the fee? Can I pay online?",
        "reference": "Fees must be paid by Demand Draft in favour of 'Institute of Innovation in Technology and Management' "
                     "payable at New Delhi. Cash payments and online transfers are not accepted.",
    },
    {
        "programme": "BBA",
        "route": "fee",
        "question": "What is the late fee if I pay 20 days after the due date?",
        "reference": "For a delay of 16 to 30 days after the due date, the late fee is Rs. 1,000 per day.",
    },
    {
        "programme": "BCA",
        "route": "fee",
        "question": "Will I get my tuition fee back if I leave after classes have started?",
        "reference": "No. If a student withdraws after classes have started and the seat stays vacant, "
                     "no tuition fee is refunded under any circumstances.",
    },
]


# 2. RUN THE ASSISTANT ON EACH QUESTION

samples = []
wrong_routes = []

for number, test in enumerate(tests, start=1):
    print(f"[{number}/{len(tests)}] {test['question']}")

    result = app.invoke({
        "programme": test["programme"],
        "messages": [("human", test["question"])],
    })

    # Check the classifier sent the question to the right PDF
    if result["query_type"] != test["route"]:
        wrong_routes.append((test["question"], test["route"], result["query_type"]))

    # Questions sent to "general" have no chunks, so RAGAS cannot score them
    if result["retrieved_context"] == "NO_RETRIEVAL_NEEDED":
        print("   -> sent to 'general', skipped\n")
        continue

    samples.append({
        "user_input": test["question"],
        "response": result["messages"][-1].content,
        # The retrieval nodes join the 4 chunks with "\n\n", so we split them back
        "retrieved_contexts": result["retrieved_context"].split("\n\n"),
        "reference": test["reference"],
    })


# 3. SCORE WITH RAGAS

print("\nScoring with RAGAS (this can take 15-20 minutes on the free Gemini plan) ...\n")

scores = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[
        Faithfulness(),
        ResponseRelevancy(strictness=1),   # 1 guessed question instead of 3, works better with Gemini
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ],
    llm=LangchainLLMWrapper(llm),                       # Gemini is the judge
    embeddings=LangchainEmbeddingsWrapper(embeddings),  # same embeddings as the app
    # max_workers=1: send one request at a time, so the Gemini free limit is not hit
    run_config=RunConfig(max_workers=1, timeout=300),
)


# 4. SHOW AND SAVE RESULTS

print("\n========== RAGAS RESULTS ==========")
print(scores)

if wrong_routes:
    print("\n========== WRONG ROUTES ==========")
    for question, expected, got in wrong_routes:
        print(f"- '{question}': expected {expected}, got {got}")
else:
    print("\nAll questions were routed to the right PDF.")

# One row per question, so you can see which answers were weak
# Saved to a new file name, so the old ragas_results.csv is not overwritten
scores.to_pandas().to_csv("ragas_results_new.csv", index=False)
print("\nSaved detailed results to ragas_results.csv")