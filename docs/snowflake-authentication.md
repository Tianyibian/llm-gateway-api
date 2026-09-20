# Snowflake authentication

The optional analytics connector supports explicit `key_pair` and `password`
authentication. Key-pair mode never falls back to a password when a key is
missing, unreadable, invalid, or protected by an incorrect passphrase.

## Prepare a local key pair

Run the following from the repository root in your own terminal. These commands
prompt for a private-key passphrase; do not paste the passphrase into chat or
include it in shell commands. Store it in a password manager.

```bash
mkdir -p .local/snowflake
chmod 700 .local/snowflake
(
  umask 077
  if [ -e .local/snowflake/rsa_key.p8 ] || [ -e .local/snowflake/rsa_key.pub ]; then
    echo "Key files already exist. Stop and inspect them; do not overwrite."
  else
    openssl genrsa 2048 | openssl pkcs8 -topk8 -v2 aes-256-cbc -inform PEM -out .local/snowflake/rsa_key.p8
    if [ $? -eq 0 ]; then
      openssl rsa -in .local/snowflake/rsa_key.p8 -pubout -out .local/snowflake/rsa_key.pub
    fi
  fi
)
```

The `.p8` file is the encrypted private key and must stay local. The `.pub` file
contains the public key that will be registered with Snowflake. Never upload the
private key to a worksheet. If generation fails, stop and inspect the error;
the existence check intentionally prevents blindly overwriting partial files.

`.gitignore` and `.dockerignore` exclude the local key directory and common
private-key extensions. Ignore rules do not protect already-tracked files.
The connector also rejects group/world-accessible private keys on POSIX systems.
Use `chmod 600` on the private key if its permissions were broadened.

## Configure the identity and backend

An administrator must register the public key with the intended user and grant
the appropriate role before the backend can connect. Prefer a dedicated service
user with only the `SMART_AI_ANALYST` role for assistant queries. Keep data-loading
credentials separate. Run `python -m app.cli.prepare_snowflake_loader` to generate
an independent encrypted loading key, private `.env.snowflake-loader`, and a
public-key-only `.local/snowflake/loader/setup_loader.sql`. An administrator must
execute that SQL before loading. The loader reads its dedicated env file and
requires `SMART_AI_LOADER_APP` with `SMART_AI_LOADER`; it refuses nonempty tables.
It does not require enabling the assistant analytics branch. Setting a role name
does not grant privileges. Do not give the loading role to the assistant user.

Keep Snowflake disabled until this setup is complete. Add the following values
to the local `.env` without changing unrelated settings:

```dotenv
SNOWFLAKE_ENABLED=false
ANALYTICS_BACKEND=neo4j
SNOWFLAKE_AUTH_METHOD=key_pair
SNOWFLAKE_ACCOUNT=your_org-your_account
SNOWFLAKE_USER=your_application_user
SNOWFLAKE_PRIVATE_KEY_FILE=.local/snowflake/rsa_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE="your_local_key_passphrase"
SNOWFLAKE_WAREHOUSE=SMART_AI_WH
SNOWFLAKE_DATABASE=SMART_AI_ANALYTICS
SNOWFLAKE_SCHEMA=ECOMMERCE
SNOWFLAKE_ROLE=SMART_AI_ANALYST
```

Relative key paths resolve from the server's working directory. An absolute
path is preferable if the server is launched from another directory. Unencrypted
PEM keys are supported by leaving the passphrase unset, but encrypted keys are
recommended. RSA keys must be at least 2048 bits.

The connector decrypts the key locally, passes PKCS#8 DER bytes through
SQLAlchemy `connect_args`, and keeps secrets out of the connection URL. Pools
are separated by account, user, role, and credential fingerprint. Replacing key
material creates a new cached engine; restart the server during planned key
rotation to close old pools and reload environment settings.

## Verification boundaries

After administrator setup, load and verify from the repository root:

```bash
.venv-langchain/bin/python -m app.cli.load_snowflake
.venv-langchain/bin/python -m app.cli.verify_snowflake
```

The loader creates no users or grants, refuses nonempty destinations, and keeps
application and loading credentials separate. The read-only verifier checks all
five table counts and all four reports against CSV for the full dataset, 2025 and
2026. It does not enable Snowflake for the running assistant.

Keep `ANALYTICS_BACKEND=neo4j` for the default graph-first prototype. To select
Snowflake reporting after successful verification, set BOTH
`SNOWFLAKE_ENABLED=true` and `ANALYTICS_BACKEND=snowflake`, then restart the server.
If Snowflake is disabled, effective analytics routing stays on Neo4j. An enabled
but failing Snowflake query reports a failure; it does not silently switch backends.
`/health` reports the effective backend.

Snowflake supports four fixed report families with absolute date bounds and a
row limit: product revenue, supplier revenue, category revenue and monthly sales.
Entity-name filters, profit, arbitrary SQL and other unsupported requests stop
before database execution. Model-based scope decisions still require evaluation;
they are not an authorization boundary.

For filesystem paths containing non-ASCII characters, use literal UTF-8 text or
relative paths in dotenv files, not JSON-style Unicode escape sequences.

`tests/test_snowflake_auth.py` tests real local key parsing and encryption, error
handling, and pool selection. Engine creation is mocked: these tests do not
authenticate to Snowflake, upload data, or consume Snowflake credits. Live
connection and analytics validation must be performed separately after setup.

References:

- [Snowflake key-pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth)
- [Snowflake SQLAlchemy key-pair support](https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#key-pair-authentication-support)
