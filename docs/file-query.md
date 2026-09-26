# Uploaded file queries

`file_query` answers questions using one document attached to the current
`POST /api/assistant` request. It is separate from shared Policy RAG, image
analysis, and the GraphRAG supervisor. No new vector or graph index is built.

## Browser test

1. Open the local chat UI and choose **File** next to the message field.
2. Select `examples/file-query-demo.txt` from this repository.
3. Ask: "Who owns Project Cedar, and when is its pilot launch?"
4. Expect the **Uploaded file** route, an answer grounded in Morgan Lee and
   October 15, 2026, and excerpt citations such as `[F1]`.
5. Reattach the same file and ask: "What is the approved project budget?"
   The answer should say the document does not provide an approved budget.

The upload is sent to the configured answer model. Do not upload credentials or
confidential material without appropriate approval. Questions and answers are
saved in the owned conversation as usual, so answers may retain quoted content.
Raw documents and extracted text are not retained as attachments or added to
shared indexes. Multipart temporary files are closed after validation. Reattach
the document for each follow-up; conversation history is not file storage.

## Request contract

Use multipart fields `query`, `user_id`, optional `conversation_id`, and `file`.
Only one document or image is allowed, never both. JSON requests remain supported;
an explicit question about an absent file routes to an attachment reminder.
`/api/classify` classifies text only and does not accept uploads.

```bash
curl -N http://127.0.0.1:8000/api/assistant \
  -F 'user_id=file-demo' \
  -F 'query=Who owns Project Cedar, and when is its pilot launch?' \
  -F 'file=@examples/file-query-demo.txt;type=text/plain'
```

SSE events include `metadata`, `route` (`file_query`), `sources`
(`backend=uploaded_file`), streamed `delta` text, and `done`. Source records
provide excerpt IDs and PDF page numbers. The prompt requests citations; it
does not prove claim-level entailment or guarantee model compliance.

## Limits and failure behavior

- UTF-8 `.txt` and `.md`, plus text-based `.pdf` only; 5 MB per file.
- At most 50 PDF pages and 30,000 extracted characters; oversized documents are
  rejected, not silently truncated. All extracted excerpts are passed to the model.
- No OCR, embedded-image reading, spreadsheet calculations, DOCX, or URL fetching.
  Text extraction may omit graphical content even in otherwise readable PDFs.
- Encrypted, malformed, binary, empty, and entirely scanned files are rejected.
- Parsing runs in a disposable child process with a 15-second wall timeout and
  CPU limit. Linux also applies an address-space limit; macOS does not. This is
  resource isolation, not a hardened sandbox for hostile production uploads.
- The multipart request is bounded before parsing. Invalid uploads are rejected
  before creating a new conversation; existing conversation ownership still applies.
- The file-answer chain has no tools. File text is untrusted human-message data,
  not a system prompt. Prompt injection resistance remains defense-in-depth, not
  a security guarantee. Production hosting needs authentication, abuse controls,
  parser isolation and explicit document retention/access policies.

## Implementation

- `app/api/assistant_uploads.py`: bounded uploads and temporary-file cleanup.
- `app/services/file_query.py`: parser worker, limits, and answer instructions.
- `app/services/assistant_service.py`: `prepare_file -> file_answer` graph branch.
- `tests/test_file_query.py`: parser, streaming, ownership, and rejection tests.

The Postman [file-query collection](../postman/File_Query.postman_collection.json)
uses the real running API. Select the sample file manually after importing.
