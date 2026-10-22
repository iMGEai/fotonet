"""User callbacks."""
from __future__ import annotations



from .options import _CALLBACK_EVENTS


def _callbacks(spec):
    """Normalize ``{event: fn | [fn, ...]}`` and reject unknown events."""
    out = {event: [] for event in _CALLBACK_EVENTS}
    for event, functions in dict(spec or {}).items():
        if event not in out:
            raise TypeError(f"Unknown callback event {event!r}; expected one of {_CALLBACK_EVENTS}.")
        out[event].extend(functions if isinstance(functions, (list, tuple)) else [functions])
    return out


def _fire(callbacks, event, payload):
    for function in callbacks.get(event, ()):
        function(payload)
