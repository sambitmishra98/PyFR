import pyfr.template as template
from pyfr.template import DottedTemplateLookup


def test_dotted_template_lookup_cache(monkeypatch):
    calls = []

    def get_data(pkg, name):
        calls.append((pkg, name))
        return b"${value}"

    monkeypatch.setattr(template.pkgutil, "get_data", get_data)
    lookup = DottedTemplateLookup("fake.pkg", {"value": "one"})

    first = lookup.get_template("kernel")
    second = lookup.get_template("kernel")

    assert first is second
    assert calls == [("fake.pkg", "kernel.mako")]
    assert first.render() == "one"

    lookup.dfltargs["value"] = "two"
    assert lookup.get_template("kernel") is first
    assert first.render() == "two"


def test_dotted_template_lookup_filter_change_invalidates(monkeypatch):
    calls = []

    def get_data(pkg, name):
        calls.append((pkg, name))
        return b"${value}"

    monkeypatch.setattr(template.pkgutil, "get_data", get_data)
    lookup = DottedTemplateLookup("fake.pkg", {"value": "one"})

    first = lookup.get_template("kernel")
    lookup.filters.append(lambda src: src.replace("value", "value"))
    second = lookup.get_template("kernel")

    assert second is not first
    assert calls == [
        ("fake.pkg", "kernel.mako"),
        ("fake.pkg", "kernel.mako"),
    ]
