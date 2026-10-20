### Q1: Why can't an LLM reliably answer questions about a private PDF that was never included in its input?

- An LLM generates responses using its learned parameters and the context available during inference. It does not automatically have access to a private PDF. To answer document-specific questions, we can use RAG: parse the document, split it into chunks, retrieve the relevant passages for a query, and include those passages in the prompt. This grounds the response in the document, although correctness still depends on retrieval quality and generation.

_The deeper technical explanation:_
An LLM has two distinct sources of information to consider:
- *Parametric knowledge*: Patterns and information encoded in the model's learned parameters during training.
- *Context supplied at inference time*: The current request, instructions, conversation history and any retrieved document passages included in the request.

#### Follow Up: If the correct passage is retrieved, is the answer guaranteed to be correct?
- No. The model may misinterpret the passage, ignore relevant information or generate unsupported claims. RAG reduces some grounding problems but does not eliminate hallucinations.

### Q2. What is the difference between a context window and long-term memory?

#### Context window

The context window is the amount of tokens a model can process within a request, subject to its particular limits and API behavior.

_It can include:_
- System instructions.
- The user's question.
- Conversation history.
- Retrieved document chunks.
- Other tool results or input content.
Depending on the model and API, generated output also consumes the available context budget.
Think of it as the model's working context for a particular inference.

#### Long-term memory

Long-term memory is information an application retains beyond an individual request and can retrieve later.
- A vector database - Stores embeddings and associated identifiers or metadata, often with text chunks, for similarity search.
- Relational or document database - Stores structured records, document contents, settings or conversation history.
- Object or file storage - Retains original PDFs and other files.

#### Follow Up: Why not put the entire PDF into the context window?

Because of context limits, token cost, latency, and the risk of irrelevant material diluting attention to important passages. For larger document collections, retrieval lets us select relevant content instead of sending everything every time.

### Q3. Follow Up: Does lowering temperature guarantee correctness?
No. Temperature affects the randomness or variability of token generation, not whether the underlying information is factually correct. A low temperature can make responses more consistent, but the model can still generate incorrect answers or confidently repeat incorrect context. To improve factual reliability in a RAG application, I would evaluate retrieval quality, ground answers in relevant passages, validate citations, define abstention behavior and test against a representative evaluation set.

### Q4. Why might we use embeddings and FAISS instead of passing all the document text directly to the LLM?

With 10,000 documents containing 100 pages each, we could have roughly a million pages of content. Passing the entire collection to an LLM for every query would likely exceed practical context limits and introduce substantial token costs and latency.
Instead, I would chunk the documents, generate embeddings for those chunks, and index the vectors using FAISS. At query time, I would embed the user's question, retrieve the top-k relevant chunks using vector similarity search, and pass only those passages to the LLM.
This reduces the amount of context sent to the model and makes query-time processing more efficient. However, it introduces new challenges: retrieval quality, chunk-size selection, index freshness, and the risk of missing relevant passages. I would measure retrieval quality and answer correctness to tune the system.

