# NN5 haftalik altin test: R motoru tahminleri sizin kendi R ciktilarinizla ayni mi? (gercek R gerekir -> Docker icinde)
Set-Location (Join-Path $PSScriptRoot "..\..")
docker compose run --rm --entrypoint python studio -m pytest tests/test_r_engine_live.py tests/test_golden_nn5.py -q -s
exit $LASTEXITCODE
