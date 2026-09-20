from fastapi.testclient import TestClient

from app.main import app


def test_chat_serves_graph_inspector_before_app_and_keeps_automatic_routing():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert html.index('/static/graph-inspector.js') < html.index('/static/app.js')
        assert 'id="graph-index-status"' in html
        assert 'Router selects the branch automatically' in html
        assert client.get('/static/graph-inspector.js').status_code == 200
        script = client.get('/static/app.js').text
        assert 'fetch("/api/assistant"' in script
        assert 'eventName === "guardrail"' in script
        assert 'eventName === "supervisor"' in script
