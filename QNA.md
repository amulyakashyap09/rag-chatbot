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

---

### Q5. Why not just split the PDF into fixed 500-character pieces straight away?
You'd lose structure (pages and sections), so you couldn't cite sources or filter. Split by structure first, then by size.

### Q6. Your RAG gives the right number with the wrong page citation. Where's the bug?
Usually in ingestion: off-by-one page numbers, or metadata that wasn't carried through chunking.

### Q7. What could go wrong in PDF parsing?
- Tables flattened into jumbled text.
- Two-column layouts read across both columns.
- Scanned pages with no text layer, which need OCR.
- Headers and footers repeated on every page.

### Q8. What precautions to be taken while ingesting the data ?
- Profiling the data before tuning the model is the kind of judgment that marks a senior candidate.
- Before building the pipeline, I profiled the corpus: page counts, the longest pages, outliers. One document was 11 pages, another was 1,000+. 
- That told me I needed metadata routing and N/A handling before tuning embeddings.

### Q9. Why did you clean the text before chunking rather than after?
- Chunk size is measured on the text, so noise makes the measurement wrong.
    - Avoid wasting tokens on unnecessary whitespace
- Chunk boundaries can't be fixed afterwards.
    - Preserve meaningful boundaries
- Overlap gets wasted the same way.
- Improve embeddings and retrieval
    - In a RAG pipeline, cleaned chunks are typically converted into embeddings and indexed for retrieval.
    - Unnecessary formatting noise can make chunks less consistent. Cleaning first helps ensure that chunking, embeddings, and indexing all operate on normalized text.

### Q10. what would break if you'd replaced <br> with \n instead of " ; "?

"Two things break. First, structure: a markdown table row must be on one line, so newlines turn table rows into loose text and the label-to-column mapping is lost. Second, chunking: splitters treat newlines as preferred split points, so newlines inside table cells invite splits between a label and its values. The chunk count barely changes. What changes is where the splits fall, and they'd fall in the worst possible place for numeric questions. ' ; ' keeps the row on one line and still reads as a separator to an LLM."

### Q11. Why chunk at all ?

- Hard limit: the embedding model only reads up to 2048 tokens per input. Anything past that is cut off or rejected.
- Retrieval precision: each chunk becomes one vector, which is roughly an average of everything in it. If a chunk covers cash flow, risk factors and a CEO letter, its vector is a blur that matches none of those topics well.
- Generation budget: the LLM sees top_k × chunk_size tokens of context. With five 2,000-token chunks, the model has to find one number in 10k tokens of text, and small models like gemma do badly at that.

### Trade-Offs B/W Small(~100 tokens) and Large(~1000+ tokens) Chunks

- Retrieval
    - Small Chunks: Sharp Vectors, precise Matches
    - Large Chunks: Blurry Vectors, Topics Mixed together

- Context
    - Small Chunks: Can lose context ("$217,901" without knowing what it measures)
    - Small Chunks: Keeps Context Together

- Cost
    - Small Chunks: Many chunks, more vectors, more to search
    - Fewer vectors, more tokens per prompt

> `No chunk size is right for every dataset. You choose one by measuring your own data, so start there.`

