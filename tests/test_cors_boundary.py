"""The local API must not opt hostile browser origins out of the SOP."""

from fastapi.testclient import TestClient

from vecgrep.backend.main import create_app


def test_foreign_origin_receives_no_cors_read_permission() -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1:8765") as client:
        response = client.get(
            "/api/health",
            headers={"Origin": "https://evil.example"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
