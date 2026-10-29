"""Small synthetic source-level evaluation using the standard evaluation scorer."""
from dataclasses import asdict
from pathlib import Path

from vecgrep.backend.config import Settings
from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.service import VecgrepService
from vecgrep.eval.gold import GoldCase
from vecgrep.eval.metrics import score_case, summarize


class FixtureEmbedder(EmbedBackend):
    name = 'ollama'
    model = 'nomic-embed-text'
    dim = 16
    terms = ('orchid', 'cobalt', 'cacheline', '雪山')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(term in text.lower()) for term in self.terms]
                + [float(not any(term in text.lower() for term in self.terms))]
                + [0.0] * 11 for text in texts]


def evaluate(root: Path) -> dict:
    rows = {}
    for backend in ('pickle', 'sqlite'):
        settings = Settings(home=root/backend/'state', qdrant_url=None, bm25_backend=backend)
        service = VecgrepService(settings)
        service._backend_cache['auto'] = FixtureEmbedder()
        docs = root/backend/'docs'; docs.mkdir(parents=True)
        bodies = ['Orchid cultivation requires filtered light and careful watering.',
                  'Cobalt alloy testing established resistance to corrosion.',
                  'Cacheline alignment improves performance of the sample allocator.',
                  '雪山天气记录。登山路线已经检查，明天早上出发。']
        for index, body in enumerate(bodies):
            (docs/f'case-{index}.md').write_text(body, encoding='utf-8')
        service.index(str(docs), 'fixture')
        for mode in ('bm25', 'vector', 'hybrid'):
            cases = [GoldCase(str(i), 'fixture', term, want=(f'case-{i}.md',))
                     for i, term in enumerate(FixtureEmbedder.terms)]
            cases.append(GoldCase('negative', 'fixture', 'unrelatedquasar', negative=True))
            scored = [score_case(case, [asdict(hit) for hit in service.search(
                case.query, 'fixture', mode=mode, top_k=3)], 0.0) for case in cases]
            rows[f'{backend}/{mode}'] = summarize(scored)
        service.store.client.close()
    return rows


def test_representative_search_fixture(tmp_path):
    for row in evaluate(tmp_path).values():
        assert row['hit@1'] == 100.0
        assert row['mrr'] == 1.0
        assert row['neg_fp_rate'] == 0.0


if __name__ == '__main__':
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(evaluate(Path(directory)), indent=2))
