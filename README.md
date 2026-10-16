# RAG on the Enron mailbox — measuring what actually works

A small Retrieval-Augmented Generation (RAG) project in TypeScript, built step by step to answer one question:

> **When does RAG actually help, and which retrieval method should it use?**

Instead of assuming that "vector search + LLM" is the right recipe, every retrieval method here is **measured** against a labelled question set, on the same data.

## TL;DR

| Retrieval method | Recall@5 (original questions) | Recall@5 (rephrased questions) |
|---|---|---|
| Semantic only (Mistral embeddings) | 77.5 % | 75.3 % |
| Keywords + IDF weighting (hand-written) | 90.1 % | 85.6 % |
| **Hybrid (keywords + semantic, RRF fusion)** | **90.7 %** | **88.2 %** |

*507-question sample, same questions for all three methods.*

- On this data, **keyword search beats pure vector search** by 10–12 points.
- Vector search is **more robust to rephrasing** (−2.2 pts vs −4.5 pts) and finds questions that keywords miss.
- **Hybrid search is the best of both**, with the largest gain on rephrased questions (+2.6 pts).
- A RAG answer costs **~1,700 tokens** per question, versus **~634,000 tokens** to send the whole mailbox to the LLM.

## Dataset

[EnronQA](https://arxiv.org/abs/2505.00263) is a research benchmark for RAG over private documents. It contains real Enron emails, each paired with questions whose answer is only in that email, plus the reference answers.

This project uses one mailbox: **`lay-k`** (Kenneth Lay, Enron CEO), `train` split.

| | |
|---|---|
| Emails | 1,317 |
| Questions | 5,570 (+ 5,570 rephrased versions) |
| Total size | ~634,000 tokens (estimated as characters / 4) |
| Average email | 482 tokens (max 1,633), so emails are indexed whole, without chunking |

Because each question was written from a single email, the expected email is known for every question. That makes retrieval **measurable**: for each question, is the right email in the top 5 results? This is the **Recall@5** metric used throughout.

## Approach

```
Question
   │
   ├──► Keyword search (IDF-weighted, hand-written) ──► top 20 ─┐
   │                                                            ├─► RRF fusion ─► top 5 emails ─┐
   └──► Semantic search (Mistral embeddings) ──────────► top 20 ─┘                              │
                                                                                                ▼
                                                     LLM (Mistral) : question + 5 emails ─► answer
```

The project was built in stages, each one measured with the same evaluation:

1. **Naive keyword search**: score = number of question words found in the email.
   → 80.7 % / 74.7 % (all 5,570 questions)
2. **IDF weighting**: rare words count more (`log(totalEmails / emailsContainingWord)`), so words like *the* or *what* score ~0.
   → 91.2 % / 88.1 % (all 5,570 questions). Errors were cut by more than half.
3. **Semantic search**: emails and questions embedded with `mistral-embed`, nearest neighbours by cosine similarity.
   → 77.5 % / 75.3 % (507-question sample)
4. **Hybrid search**: both rankings merged with **Reciprocal Rank Fusion** (`1 / (60 + rank)`), which uses positions rather than scores.
   → 90.7 % / 88.2 % (507-question sample)
5. **Generation**: the 5 retrieved emails and the question go to an LLM, which is instructed to answer **only** from those emails and to cite the email id.

### Why keywords win here

EnronQA questions target precise details: names, dates, phone numbers. Rare-word matching is very strong on that. An embedding compresses a whole email into one vector, so a specific detail is diluted, and many of Kenneth Lay's emails (meetings, travel, scheduling) end up close to each other in vector space.

Semantic search still matters: it finds 8 (original) and 17 (rephrased) questions that keyword search misses. That complementarity is what makes the hybrid worth it.

## Project structure

| File | Role |
|---|---|
| `explorer.ts` | Keyword search (naive, then IDF) + Recall@5 evaluation |
| `embed.ts` | Computes and saves embeddings for all emails and the question sample (run once) |
| `semantic-eval.ts` | Recall@5 of semantic search, from the saved vectors |
| `hybrid.ts` | Compares keywords, semantic and hybrid on the same sample |
| `search.ts` | Reusable hybrid search module (`hybridSearch`) |
| `rag.ts` | Full RAG: hybrid retrieval + LLM answer, interactive in the terminal |
| `baseline.ts` | Sends the emails directly to the LLM, without RAG, to compare token usage |

## Setup

Requirements: **Node.js 23.6+** (runs `.ts` files natively), [DuckDB CLI](https://duckdb.org/docs/installation), a [Mistral API key](https://console.mistral.ai) (the free tier is enough).

```bash
npm install
```

**1. Export the mailbox from Hugging Face**

```bash
duckdb -c "
COPY (
  SELECT path, email, questions, rephrased_questions, gold_answers
  FROM 'hf://datasets/MichaelR207/enron_qa_0922@~parquet/default/train/*.parquet'
  WHERE user = 'lay-k'
) TO 'lay-k.json' (FORMAT JSON, ARRAY true);
"
```

**2. Add your API key**

```bash
cp .env.example .env   # then put your Mistral key in .env
```

**3. Run**

```bash
node --env-file=.env explorer.ts       # keyword search evaluation (no API needed)
node --env-file=.env embed.ts          # compute embeddings once (~3–4 min on the free tier)
node --env-file=.env hybrid.ts         # compare the three retrieval methods
node --env-file=.env rag.ts            # ask your own questions
```

The free Mistral tier has strict rate limits. API calls are sent in small batches with pauses, and retried with backoff on `429` errors.

## Limitations

- **One mailbox, one split.** Results are for `lay-k` / `train` only.
- **Sample size.** Semantic and hybrid results use 507 of the 5,570 questions. Small gaps (such as +0.6 pt for hybrid on original questions, i.e. 3 questions) are within sampling noise; the +2.6 pt gain on rephrased questions is more meaningful.
- **Retrieval only is evaluated.** The quality of the final LLM answers has not been measured yet.
- **Token counts are estimated** (characters / 4), except where the API returns usage.

## Next steps

- Evaluate final answers against the reference answers (LLM-as-a-judge).
- Complete BM25 (term-frequency saturation, length normalisation) and compare with the IDF-only version.
- Run the full 5,570 questions for semantic and hybrid search.
- Store vectors in PostgreSQL with **pgvector** and use its full-text search for keywords.
- Compare with an **agentic** approach, where the LLM calls the search as a tool, on accuracy, tokens and latency.

## References

- *EnronQA: Towards Personalized RAG over Private Documents*, 2025: [arXiv:2505.00263](https://arxiv.org/abs/2505.00263) · [dataset](https://huggingface.co/datasets/MichaelR207/enron_qa_0922)
- Anthropic, [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
