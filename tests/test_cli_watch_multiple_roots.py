from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from watchfiles import Change

from vecgrep.cli import main as m


def test_watch_indexes_and_follows_both_roots(tmp_path: Path):
    roots = [tmp_path / 'local', tmp_path / 'desktop']
    for root in roots:
        root.mkdir()
        (root / 'session.md').write_text('new session content')
    indexed = []
    watched = []

    def events(*paths, **kwargs):
        watched.extend(paths)
        yield {(Change.modified, str(root / 'session.md')) for root in roots}
        yield {(Change.modified, str(roots[1] / 'state.json'))}

    m._WATCH_SEEN_HASHES.clear()
    m._WATCH_PENDING.clear()
    with patch('watchfiles.watch', events), patch.object(
        m, '_do_index', side_effect=lambda source, *a, **kw: indexed.append(source)
    ):
        result = CliRunner().invoke(m.cli, ['watch', str(roots[0]), '--also-watch',
            str(roots[1]), '--corpus', 'test', '--include', '*.md', '--quiet-period', '0'])
    assert result.exit_code == 0, result.output
    assert watched == list(map(str, roots))
    assert indexed[:2] == list(map(str, roots))
    assert set(indexed[2:]) == {str(root / 'session.md') for root in roots}


def test_watch_validates_all_roots_before_indexing(tmp_path: Path):
    with patch.object(m, '_do_index') as index:
        result = CliRunner().invoke(m.cli, ['watch', str(tmp_path), '--also-watch',
            str(tmp_path / 'missing'), '--corpus', 'test'])
    assert result.exit_code != 0
    assert 'must be a directory' in result.output
    index.assert_not_called()


def test_initial_failure_does_not_skip_other_root(tmp_path: Path):
    other = tmp_path / 'desktop'
    other.mkdir()
    with patch('watchfiles.watch', return_value=iter(())), patch.object(
        m, '_do_index', side_effect=[TimeoutError('temporary'), None]
    ) as index:
        result = CliRunner().invoke(m.cli, ['watch', str(tmp_path), '--also-watch',
            str(other), '--corpus', 'test'])
    assert result.exit_code == 0, result.output
    assert [call.args[0] for call in index.call_args_list] == [str(tmp_path), str(other)]
    assert 'watching anyway' in result.output
