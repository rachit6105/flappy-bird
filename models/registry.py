import difflib
import functools
import inspect

MODELS: dict[str, type] = {}


class UserError(Exception):
    """An error that should be shown to the user without a traceback."""


class ModelNotFound(UserError):
    pass


def _record_init_kwargs(cls):
    """Wrap __init__ so every instance remembers the kwargs it was built with.

    This is what makes checkpoints self-describing without training code
    having to pass the constructor args around by hand.
    """
    orig = cls.__init__
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if type(self) is not cls:  # ignore super().__init__ calls from subclasses
            return
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        cfg = {}
        for pname, value in list(bound.arguments.items())[1:]:
            kind = sig.parameters[pname].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                cfg.update(value)
            elif kind is not inspect.Parameter.VAR_POSITIONAL:
                cfg[pname] = value
        self.init_kwargs = cfg

    cls.__init__ = __init__


def register_model(cls=None, *, history: bool = False):
    """Register a policy as '<module>.<ClassName>', e.g. 'gru.GRU1'.

    history=True means the model consumes a stack of K frames (MLP_Temp*).
    Usable as @register_model or @register_model(history=True).
    """
    def deco(cls):
        family = cls.__module__.rsplit(".", 1)[-1]
        name = f"{family}.{cls.__name__}"
        if name in MODELS and MODELS[name] is not cls:
            raise ValueError(f"Duplicate model name: {name}")
        cls.registry_name = name
        cls.uses_history = history
        _record_init_kwargs(cls)
        MODELS[name] = cls
        return cls

    return deco if cls is None else deco(cls)


def family_of(name: str) -> str:
    return name.split(".", 1)[0]


def families() -> list[str]:
    return sorted({family_of(n) for n in MODELS})


def models_in(family: str) -> list[str]:
    return sorted(n for n in MODELS if family_of(n) == family)


def get_model(name: str) -> type:
    lowered = {n.lower(): n for n in MODELS}
    if name.lower() in lowered:
        return MODELS[lowered[name.lower()]]
    hint = difflib.get_close_matches(name, MODELS, n=1, cutoff=0.5)
    msg = f"Unknown model '{name}'."
    if hint:
        msg += f" Did you mean '{hint[0]}'?"
    raise ModelNotFound(msg + " Run `flappy-bird list` to see available models.")