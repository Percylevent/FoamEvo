"""Deterministic applicability predicates; no inferred solver-family compatibility."""
import json
import math
import os

UNKNOWN = {'', 'unknown', 'none', 'null'}

def _value(value):
    return str(value).strip().lower() if value is not None else ''

def _matches(declared, actual):
    if declared is None or declared == '' or _value(declared) == 'any':
        return True
    if isinstance(declared, list):
        return bool(declared) and any(_matches(x, actual) for x in declared)
    if isinstance(declared, dict):
        if not declared or not set(declared) <= {'min', 'max'} or isinstance(actual, bool):
            return False
        try:
            x = float(actual)
            low, high = float(declared.get('min', '-inf')), float(declared.get('max', 'inf'))
            return math.isfinite(x) and not math.isnan(low) and not math.isnan(high) and low <= x <= high
        except (TypeError, ValueError):
            return False
    return _value(actual) not in UNKNOWN and _value(actual) == _value(declared)

def valid_conditions(conditions):
    if not isinstance(conditions, dict):
        return False
    def scalar(value):
        return (isinstance(value, (str, int, float, bool)) and str(value).strip() != ""
                and (not isinstance(value, float) or math.isfinite(value)))
    for key, declared in conditions.items():
        if not isinstance(key, str) or not key.strip():
            return False
        if isinstance(declared, dict):
            if not declared or not set(declared) <= {"min", "max"}:
                return False
            try:
                if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in declared.values()):
                    return False
                if float(declared.get("min", "-inf")) > float(declared.get("max", "inf")):
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(declared, list):
            if not declared or not all(scalar(x) for x in declared):
                return False
        elif not scalar(declared):
            return False
    return True


def applicability(metadata, *, solver='', fork='', version='', context=None):
    """Return (accepted, rejection reason); missing constrained values fail closed."""
    if _value(metadata.get('status', 'active')) != 'active':
        return False, 'inactive'
    for key, actual in (('solver', solver), ('fork', fork), ('version', version)):
        declared = metadata.get(key)
        if _value(declared) == 'unknown' or not _matches(declared, actual):
            return False, 'incompatible_or_unknown_' + key
    conditions = metadata.get('conditions', {})
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions or '{}')
        except ValueError:
            return False, 'malformed_conditions'
    if not valid_conditions(conditions):
        return False, 'malformed_conditions'
    actual_context = context or {}
    if not isinstance(actual_context, dict):
        return False, 'invalid_context'
    for key, declared in conditions.items():
        if not _matches(declared, actual_context.get(key)):
            return False, 'unmet_condition:' + key
    return True, ''

def task_context(config=None):
    value = getattr(config, 'experience_context', None) if config else None
    if value is None:
        try:
            value = json.loads(os.getenv('FOAM_RUNNER_EXPERIENCE_CONTEXT', '{}'))
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}

