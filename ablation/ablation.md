| Row | Change relative to the row above | Recall@5 | Kept | Rejected | Fact | Table | Comparison | Unanswerable | Accuracy | Points 2.1-4 (of 50) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | Naive RAG: get_text(), RecursiveCharacterTextSplitter, whitespace BM25 top-5 over the corpus, never [] | 34.8% | 100.0% | 0.0% | 67% | 23% | 33% | 86% | 48.3% | 14.67 |
| V1 | Tables through table_to_text | 37.0% | 100.0% | 0.0% | 67% | 27% | 25% | 100% | 51.7% | 15.33 |
| V2 | Chunks follow the headings instead of fixed-size pieces | 23.9% | 100.0% | 0.0% | 67% | 9% | 17% | 93% | 41.7% | 13.33 |
| V3 | Contextual chunk header (company, document, section) | 60.9% | 100.0% | 0.0% | 75% | 45% | 83% | 86% | 68.3% | 22.74 |
| V4 | Tokeniser: no punctuation, light stemming, British spelling | 73.9% | 100.0% | 0.0% | 92% | 55% | 92% | 86% | 76.7% | 29.30 |
| V5 | Query: content words, synonyms, abbreviations | 84.8% | 100.0% | 0.0% | 100% | 91% | 58% | 86% | 85.0% | 35.04 |
| V6 | parse_query and metadata filter (company, year, document type) | 100.0% | 100.0% | 0.0% | 100% | 100% | 100% | 100% | 100.0% | 40.00 |
| V7 | Comparisons: the companies take turns | 100.0% | 100.0% | 0.0% | 100% | 100% | 92% | 100% | 98.3% | 39.67 |
| V8 | Table chunks x1.3 when the question asks for or compares figures | 100.0% | 100.0% | 0.0% | 100% | 100% | 100% | 100% | 100.0% | 40.00 |
| V9 | Abstention: unknown company, uncovered year, unknown word + LLM check | 100.0% | 100.0% | 100.0% | 100% | 100% | 100% | 100% | 100.0% | 50.00 |
