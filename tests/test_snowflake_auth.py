from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.services.errors import LLMConfigurationError
from app.services.snowflake_analytics import (
    SnowflakeAnalyticsService,
    _SNOWFLAKE_ENGINES,
    _authentication_parameters,
    create_snowflake_engine,
    dispose_snowflake_engines,
)


@pytest.fixture
def settings(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith("SNOWFLAKE_"):
            monkeypatch.delenv(name)
    return Settings(
        _env_file=None,
        snowflake_enabled=True,
        snowflake_account="test-org",
        snowflake_user="test-user",
        snowflake_warehouse="TEST_WH",
        snowflake_database="TEST_DB",
        snowflake_schema="TEST_SCHEMA",
        snowflake_role="TEST_READER",
        snowflake_auth_method="key_pair",
    )


@pytest.fixture
def key_writer(tmp_path):
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")

    def write(passphrase=None, bits=2048):
        key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
        path = tmp_path / "test-key.p8"
        path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(passphrase) if passphrase else serialization.NoEncryption(),
        ))
        path.chmod(0o600)
        der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        return path, der

    return write


@pytest.fixture
def engine_factory(monkeypatch):
    pytest.importorskip("snowflake.sqlalchemy")
    factory = Mock(side_effect=lambda *args, **kwargs: Mock())
    monkeypatch.setattr("sqlalchemy.create_engine", factory)
    dispose_snowflake_engines()
    yield factory
    dispose_snowflake_engines()


@pytest.mark.parametrize("passphrase", [None, b"local-test-passphrase"])
def test_key_pair_uses_der_connect_args_without_password(settings, key_writer, engine_factory, passphrase):
    path, der = key_writer(passphrase)
    settings.snowflake_private_key_file = str(path)
    settings.snowflake_private_key_passphrase = SecretStr(passphrase.decode()) if passphrase else None
    settings.snowflake_password = SecretStr("must-not-be-used")

    SnowflakeAnalyticsService.from_settings(settings)

    args, kwargs = engine_factory.call_args
    assert "must-not-be-used" not in str(args[0])
    assert str(path) not in str(args[0])
    assert kwargs["connect_args"]["private_key"] == der
    assert kwargs["connect_args"]["authenticator"] == "SNOWFLAKE_JWT"
    assert "password" not in kwargs["connect_args"]
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["connect_args"]["session_parameters"]["STATEMENT_TIMEOUT_IN_SECONDS"] == 30
    assert all(der not in cache_key for cache_key in _SNOWFLAKE_ENGINES)


def test_explicit_password_mode_remains_supported(settings, engine_factory):
    settings.snowflake_auth_method = "password"
    settings.snowflake_password = SecretStr("legacy-test-secret")
    create_snowflake_engine(settings)
    args, kwargs = engine_factory.call_args
    assert "legacy-test-secret" not in str(args[0])
    assert kwargs["connect_args"]["password"] == "legacy-test-secret"
    assert "private_key" not in kwargs["connect_args"]
    assert all("legacy-test-secret" not in key for key in _SNOWFLAKE_ENGINES)


def test_missing_key_does_not_fall_back_to_password(settings):
    settings.snowflake_password = SecretStr("do-not-fallback")
    with pytest.raises(LLMConfigurationError, match="SNOWFLAKE_PRIVATE_KEY_FILE"):
        _authentication_parameters(settings)


def test_missing_password_is_explicit(settings):
    settings.snowflake_auth_method = "password"
    with pytest.raises(LLMConfigurationError, match="SNOWFLAKE_PASSWORD"):
        _authentication_parameters(settings)


@pytest.mark.parametrize("case", ["missing_file", "invalid_pem", "wrong_passphrase"])
def test_key_errors_are_sanitized(settings, key_writer, tmp_path, case):
    path, _ = key_writer(b"correct-test-passphrase")
    if case == "missing_file":
        path = tmp_path / "missing-key.p8"
    elif case == "invalid_pem":
        path.write_text("private-invalid-test-material")
    settings.snowflake_private_key_file = str(path)
    settings.snowflake_private_key_passphrase = SecretStr("wrong-test-passphrase")
    with pytest.raises(LLMConfigurationError, match="Unable to load") as caught:
        _authentication_parameters(settings)
    assert str(path) not in str(caught.value)
    assert "wrong-test-passphrase" not in str(caught.value)
    assert "private-invalid-test-material" not in str(caught.value)


def test_broad_private_key_permissions_are_rejected(settings, key_writer):
    import os

    if os.name != "posix":
        pytest.skip("POSIX permissions only")
    path, _ = key_writer()
    path.chmod(0o644)
    settings.snowflake_private_key_file = str(path)
    with pytest.raises(LLMConfigurationError, match="chmod 600"):
        _authentication_parameters(settings)


def test_undersized_rsa_key_is_rejected(settings, key_writer):
    path, _ = key_writer(bits=1024)
    settings.snowflake_private_key_file = str(path)
    with pytest.raises(LLMConfigurationError, match="at least 2048"):
        _authentication_parameters(settings)


def test_cache_reuses_identity_but_separates_role_and_rotated_keys(settings, key_writer, engine_factory):
    path, _ = key_writer()
    settings.snowflake_private_key_file = str(path)
    first = create_snowflake_engine(settings)
    assert create_snowflake_engine(settings) is first
    assert create_snowflake_engine(settings, role="TEST_LOADER") is not first
    key_writer()  # Rotate key material at the same path.
    assert create_snowflake_engine(settings) is not first
    assert engine_factory.call_count == 3
    dispose_snowflake_engines()
    first.dispose.assert_called_once()
    assert not _SNOWFLAKE_ENGINES


def test_data_loader_engine_also_validates_connection_fields(settings):
    settings.snowflake_account = " "
    with pytest.raises(LLMConfigurationError, match="SNOWFLAKE_ACCOUNT"):
        create_snowflake_engine(settings)


def test_unknown_auth_mode_rejected_by_settings():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, snowflake_auth_method="guess")
