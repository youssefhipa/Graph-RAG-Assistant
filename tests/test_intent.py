import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(ROOT))

from app.intent import IntentClassifier  # noqa: E402


def test_basic_intents():
    clf = IntentClassifier()
    assert clf.predict("find electronics in SP").intent == "product_search"
    assert clf.predict("why is delivery late in RJ").intent == "delivery_delay"
    assert clf.predict("seller reliability in MG").intent == "seller_performance"
    assert clf.predict("state trend for RJ").intent in {"state_trend", "product_search"}
