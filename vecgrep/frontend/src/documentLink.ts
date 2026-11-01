/** The backend still validates a linked source against the selected corpus. */
export function documentLink(search: string): { corpus: string; sourceId: string; nonce: number } | null {
  const params = new URLSearchParams(search);
  const corpus = params.get("corpus");
  const sourceId = params.get("source");
  return corpus && sourceId ? { corpus, sourceId, nonce: 1 } : null;
}
