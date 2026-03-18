RAG Injection Guard

A lightweight agent that protects RAG pipelines from prompt injection by filtering unsafe knowledge base.

Approach

The system follows a simple and deterministic pipeline.
First, it retrieves relevant chunks from the knowledge base.
Then, it ranks them using keyword overlap with the user query and keeps only the most relevant ones.
Each chunk is scanned for known injection patterns such as instruction overrides, role hijacking, tool execution attempts, and data exfiltration.
Unsafe chunks are dropped and logged as security events.
The final response is generated only from the remaining trusted content.
If no safe content is left, the system returns a fixed safe fallback response instead of generating an answer.

Key decisions

A rule based detector is used instead of a model based classifier to ensure predictable behavior and low latency.
The system enforces a strict two outcome policy so it either answers from trusted data or blocks completely.
Chunk level filtering is applied before generation to prevent poisoned content from influencing the model.
Security events are logged but not exposed to the end user to avoid leaking detection logic.

Tradeoffs

Strict filtering improves safety but can reduce recall if legitimate content is mistakenly flagged.
The two outcome simplifies behavior but limits flexibility in partial or uncertain scenarios.
Keyword based ranking is efficient but less semantically rich than embedding based retrieval.
Keeping the system dependency free improves reliability but avoids more advanced techniques that require external libraries.
