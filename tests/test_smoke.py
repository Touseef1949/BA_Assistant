import importlib


def test_core_modules_import_and_constants_exist():
    app = importlib.import_module("app")
    payment = importlib.import_module("payment")
    preflight = importlib.import_module("preflight")

    assert app.TEXT_ANALYSIS_MODEL_ID == "deepseek-v4-flash"
    assert app.APP_URL == "https://touseefshaik.com"
    assert "Standard" in app.ANALYSIS_TYPES
    assert payment.FREE_USAGE_LIMIT == 2
    assert payment.OTP_TTL_MINUTES > 0
    assert callable(preflight.check_syntax)


def test_load_test_scaffold_targets_only_safe_endpoints():
    with open("tests/load/locustfile.py", "r", encoding="utf-8") as handle:
        locustfile = handle.read()

    assert '"/"' in locustfile
    assert '"/_stcore/health"' in locustfile
    assert "Generate BA Report" not in locustfile
    assert "analysis" not in locustfile.lower()
