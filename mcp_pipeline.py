import json, uuid, datetime
from transformers import pipeline

# initialize local summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# define session and context (like MCP)
session = {
    "session_id": str(uuid.uuid4()),
    "created_at": datetime.datetime.now().isoformat(),
    "context": {"task": "summarization", "domain": "nlp"},
    "actions": []
}

with open("data/article.txt") as f:
    text = f.read()

# log reasoning step
action = {
    "step": 1,
    "type": "summarization",
    "input_length": len(text),
    "timestamp": datetime.datetime.now().isoformat()
}
session["actions"].append(action)

summary = summarizer(text, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]

session["actions"][-1]["output_length"] = len(summary)
session["actions"][-1]["output"] = summary

print("🔹 MCP-Style Summary:\n", summary)

# store structured trace
with open("logs/session_trace.json", "w") as f:
    json.dump(session, f, indent=2)
