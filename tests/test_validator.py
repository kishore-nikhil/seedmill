"""Dynamic pydantic model construction and record serialization."""

from __future__ import annotations

from seedmill.validator import build_pydantic_model, validate_single


def _model(fields, name="rec"):
    return build_pydantic_model({"name": name, "fields": fields})


def test_required_field_missing_returns_an_error():
    data, err = validate_single({}, _model({"a": {"type": "str"}}))
    assert data is None
    assert isinstance(err, str) and err


def test_enum_value_serializes_to_a_raw_string():
    """
    The contract every downstream consumer depends on: exporters, hooks
    and the JSONL writer all expect plain values, never Enum members.
    """
    model = _model({"color": {"type": "str", "enum": ["blue", "red"]}})
    data, err = validate_single({"color": "blue"}, model)
    assert err is None
    assert data == {"color": "blue"}
    assert type(data["color"]) is str


def test_enum_rejects_an_out_of_vocabulary_value():
    model = _model({"color": {"type": "str", "enum": ["blue", "red"]}})
    data, err = validate_single({"color": "chartreuse"}, model)
    assert data is None and err


def test_nullable_field_absent_comes_back_none():
    data, err = validate_single({}, _model({"note": {"type": "str", "nullable": True}}))
    assert err is None and data == {"note": None}


def test_default_is_applied_when_absent():
    data, err = validate_single({}, _model({"n": {"type": "int", "default": 7}}))
    assert err is None and data == {"n": 7}


def test_nested_object_round_trips_to_a_plain_dict():
    model = _model(
        {
            "cfg": {
                "type": "object",
                "fields": {"n": {"type": "int"}, "kind": {"type": "str", "enum": ["x", "y"]}},
            }
        }
    )
    data, err = validate_single({"cfg": {"n": 3, "kind": "x"}}, model)
    assert err is None
    assert data == {"cfg": {"n": 3, "kind": "x"}}
    assert isinstance(data["cfg"], dict)


def test_nullable_nested_object_accepts_null():
    model = _model(
        {"cfg": {"type": "object", "nullable": True, "fields": {"n": {"type": "int"}}}}
    )
    data, err = validate_single({"cfg": None}, model)
    assert err is None and data == {"cfg": None}


def test_list_of_enum_items_serializes_to_raw_strings():
    model = _model({"tags": {"type": "list", "items": {"type": "str", "enum": ["a", "b"]}}})
    data, err = validate_single({"tags": ["a", "b"]}, model)
    assert err is None
    assert data == {"tags": ["a", "b"]}
    assert all(type(t) is str for t in data["tags"])


def test_list_of_enum_items_rejects_an_out_of_vocabulary_item():
    model = _model({"tags": {"type": "list", "items": {"type": "str", "enum": ["a", "b"]}}})
    data, err = validate_single({"tags": ["a", "zzz"]}, model)
    assert data is None and err


def test_numeric_string_coercion_is_pinned():
    """
    Pins what pydantic v2 currently does with "5" for an int field, so a
    pydantic upgrade that changes it surfaces as a test failure rather
    than as silently different generated data.
    """
    data, err = validate_single({"n": "5"}, _model({"n": {"type": "int"}}))
    assert err is None
    assert data == {"n": 5} and type(data["n"]) is int


def test_model_name_derives_from_the_schema_name():
    model = build_pydantic_model({"name": "icon classifier", "fields": {"a": {"type": "str"}}})
    assert model.__name__ == "Icon_Classifier"
