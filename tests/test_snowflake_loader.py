from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.cli import load_snowflake as loader
from app.cli import prepare_snowflake_loader as preparation
from app.core.config import Settings
from app.services.errors import LLMConfigurationError
from app.services.snowflake_analytics import _authentication_parameters


@pytest.fixture(params=["workspace", "workspace-\u627e\u5de5\u4f5c"])
def prepared(tmp_path, monkeypatch, request):
    import os
    for name in list(os.environ):
        if name.startswith("SNOWFLAKE_"):
            monkeypatch.delenv(name)
    template = Path("snowflake/setup_loader.sql").read_text()
    tmp_path = tmp_path / request.param
    tmp_path.mkdir()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "snowflake").mkdir()
    (tmp_path / "snowflake/setup_loader.sql").write_text(template)
    settings = Settings(
        _env_file=None, snowflake_account="test-account", snowflake_user="SMART_AI_APP",
        snowflake_database="SMART_AI_ANALYTICS", snowflake_schema="ECOMMERCE",
        snowflake_warehouse="SMART_AI_WH",
    )
    monkeypatch.setattr(preparation, "get_settings", lambda: settings)
    sql, env = preparation.prepare()
    return sql, env


def test_preparation_separates_loader_identity_and_keeps_secrets_local(prepared):
    sql, env = prepared
    settings = loader.loader_settings(env)
    assert settings.snowflake_user == "SMART_AI_LOADER_APP"
    assert settings.snowflake_enabled is False
    assert _authentication_parameters(settings)["authenticator"] == "SNOWFLAKE_JWT"
    assert settings.snowflake_private_key_passphrase.get_secret_value() not in sql.read_text()
    assert "PRIVATE KEY" not in sql.read_text()
    assert "PASTE_LOADER_PUBLIC_KEY_HERE" not in sql.read_text()
    assert not Path(".env").exists()
    assert env.stat().st_mode & 0o777 == 0o600
    assert Path(settings.snowflake_private_key_file).stat().st_mode & 0o777 == 0o600


def test_preparation_refuses_to_overwrite_credentials(prepared):
    _, env = prepared
    original = env.read_bytes()
    with pytest.raises(LLMConfigurationError, match="already exist"):
        preparation.prepare()
    assert env.read_bytes() == original


def test_inherited_app_identity_cannot_be_used_for_loading(prepared, monkeypatch):
    _, env = prepared
    monkeypatch.setenv("SNOWFLAKE_USER", "SMART_AI_APP")
    with pytest.raises(LLMConfigurationError, match="separate"):
        loader.loader_settings(env)


def test_loader_requires_explicit_env_file(tmp_path):
    with pytest.raises(LLMConfigurationError, match="dedicated"):
        loader.loader_settings(tmp_path / "missing.env")


def test_source_validation_preserves_all_columns_and_files(tmp_path):
    content = 'ID,ContactName,Address\n1,"Example, Person",Demo address\n'
    for filename in loader.TABLE_FILES.values():
        (tmp_path / filename).write_text(content)
    sources = loader.read_sources(tmp_path)
    assert len(sources) == 5
    for path, count, headers in sources.values():
        assert count == 1
        assert headers == ["ID", "CONTACTNAME", "ADDRESS"]
        assert path.read_text() == content


@pytest.mark.parametrize("content", ["ID,ID\n1,2\n", "ID,Name\n1\n", "ID,Name\n"])
def test_bad_sources_fail_before_connection(tmp_path, content):
    (tmp_path / "Categories.csv").write_text(content)
    with pytest.raises(ValueError):
        loader.read_sources(tmp_path)


@pytest.mark.parametrize("failure", [None, "nonempty", "columns", "copy_count"])
def test_upload_checks_and_transaction_scope(monkeypatch, tmp_path, capsys, failure):
    sources = {table: (tmp_path / filename, 1, ["ID", "NAME"]) for table, filename in loader.TABLE_FILES.items()}
    monkeypatch.setattr(loader, "loader_settings", lambda _: Settings(_env_file=None))
    monkeypatch.setattr(loader, "read_sources", lambda _: sources)
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    engine.begin.return_value.__enter__.return_value = connection
    monkeypatch.setattr(loader, "create_snowflake_engine", lambda _: engine)
    dispose = MagicMock()
    monkeypatch.setattr(loader, "dispose_snowflake_engines", dispose)
    copied = set()
    calls = []

    def execute(sql):
        calls.append(sql)
        result = MagicMock()
        if sql.startswith("SELECT CURRENT_USER"):
            result.one.return_value = ("SMART_AI_LOADER_APP", "SMART_AI_LOADER", "SMART_AI_ANALYTICS", "ECOMMERCE")
        elif sql.startswith("SELECT COUNT"):
            table = sql.split()[-1]
            result.scalar_one.return_value = int(table in copied or failure == "nonempty")
        elif sql.startswith("DESC"):
            result.mappings.return_value.all.return_value = [{"name": name} for name in (["WRONG"] if failure == "columns" else ["ID", "NAME"])]
        elif sql.startswith("PUT"):
            table = sql.split("/ ")[0].split("/")[-1]
            result.mappings.return_value.all.return_value = [{"status": "UPLOADED", "target": sources[table][0].name + ".gz"}]
        elif sql.startswith("COPY"):
            table = sql.split()[2]
            copied.add(table)
            result.mappings.return_value.all.return_value = [{"status": "LOADED", "rows_loaded": 0 if failure == "copy_count" else 1}]
        else:
            raise AssertionError(sql)
        return result

    connection.exec_driver_sql.side_effect = execute
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            loader.load()
        assert "committed" not in capsys.readouterr().out
        if failure in {"nonempty", "columns"}:
            assert not any(sql.startswith("PUT") for sql in calls)
            engine.begin.assert_not_called()
        else:
            assert engine.begin.return_value.__exit__.call_args.args[0] is RuntimeError
    else:
        loader.load()
        assert len(copied) == 5
        engine.begin.assert_called_once()
        assert engine.begin.return_value.__exit__.call_args.args[0] is None
        assert "All five tables committed" in capsys.readouterr().out
    assert not any("TRUNCATE" in sql or "FORCE=TRUE" in sql or "@%" in sql for sql in calls)
    dispose.assert_called_once()
    engine.update_execution_options.assert_called_once_with(autocommit=False)
