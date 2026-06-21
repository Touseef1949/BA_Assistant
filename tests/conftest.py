"""Force REQUIRE_AUTH to True during test runs by reimporting payment fresh."""
import os
import sys

os.environ["REQUIRE_AUTH"] = "true"

# Force-reload payment module so REQUIRE_AUTH picks up the env var
for mod in list(sys.modules):
    if mod in ("payment", "app"):
        del sys.modules[mod]
