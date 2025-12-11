"""
brmspy.types.formula_dsl
"""

from dataclasses import dataclass
from collections.abc import Iterator
from typing import Literal, Mapping, Sequence, Union, cast, get_args

from .brms_results import ProxyListSexpVector
from rpy2.rinterface_lib.sexp import Sexp


__all__ = ["Primitive", "FormulaPart", "FormulaConstruct", "Node"]

Primitive = Union[
    int, float, str, bool, None, "FormulaConstruct", "FormulaPart", ProxyListSexpVector
]

_FORMULA_FUNCTION_WHITELIST = Literal[
    "bf",
    "lf",
    "nlf",
    "acformula",
    "set_rescor",
    "set_mecor",
    "set_nl",
]


@dataclass
class FormulaPart:
    _fun: _FORMULA_FUNCTION_WHITELIST
    _args: Sequence[Primitive]
    _kwargs: Mapping[str, Primitive]

    def __post_init__(self):
        # Validate function name first
        if self._fun not in get_args(_FORMULA_FUNCTION_WHITELIST):
            raise ValueError(
                f"FormulaPart._fun must be one of {_FORMULA_FUNCTION_WHITELIST!r}, "
                f"got {self._fun!r}"
            )

        # Enforce _args is a list
        if not isinstance(self._args, list):
            raise TypeError(
                f"FormulaPart._args must be a list, got {type(self._args).__name__}"
            )

        # Enforce _kwargs is a dict
        if not isinstance(self._kwargs, dict):
            raise TypeError(
                f"FormulaPart._kwargs must be a dict, got {type(self._kwargs).__name__}"
            )

    def __str__(self) -> str:
        args = ", ".join(repr(a) for a in self._args)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        inner = ", ".join(x for x in (args, kwargs) if x)
        return f"{self._fun}({inner})"

    def __repr__(self) -> str:
        return self.__str__()


Node = FormulaPart | ProxyListSexpVector | list["Node"]
Other = Union[str, "FormulaConstruct", FormulaPart, ProxyListSexpVector]
Summand = tuple[FormulaPart | ProxyListSexpVector, ...]


def _sexp_to_str(o: ProxyListSexpVector) -> str:
    s = repr(o)

    if "family:" not in s.lower():
        return ""

    family: str | None = None
    link: str | None = None

    for raw_line in s.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if lower.startswith("family:"):
            # text after the colon
            family = line.split(":", 1)[1].strip() or None

        elif lower.startswith("link function:"):
            link = line.split(":", 1)[1].strip() or None

    if family is None:
        return ""

    # identity is the default; don't bother printing it
    if link is None or link.lower() == "identity":
        return f"{family}()"

    return f"{family}(link='{link}')"


@dataclass
class FormulaConstruct:
    _parts: list[Node]

    @classmethod
    def _formula_parse(cls, obj: Other) -> "FormulaConstruct":
        if isinstance(obj, FormulaConstruct):
            return obj
        if isinstance(obj, ProxyListSexpVector):
            return FormulaConstruct(_parts=[obj])
        if isinstance(obj, FormulaPart):
            return FormulaConstruct(_parts=[obj])
        if isinstance(obj, str):
            # You’ll hook this into your real DSL later
            part = FormulaPart(_fun="bf", _args=[obj], _kwargs={})
            return FormulaConstruct(_parts=[part])
        raise TypeError(
            f"Cannot parse object of type {type(obj)!r} into FormulaConstruct"
        )

    def __add__(self, other: Other):

        if isinstance(other, (FormulaPart, str, ProxyListSexpVector)):
            other = FormulaConstruct._formula_parse(other)

        if not isinstance(other, FormulaConstruct):
            raise ArithmeticError(
                "When adding values to formula, they must be FormulaResult or parseable to FormulaResult"
            )

        if len(other._parts) <= 1:
            return FormulaConstruct(_parts=self._parts + other._parts)
        else:
            return FormulaConstruct(_parts=[self._parts, other._parts])

    def __radd__(self, other: Other) -> "FormulaConstruct":
        # To support: "y ~ x" + something
        return self._formula_parse(other) + self

    def iter_summands(self) -> Iterator[Summand]:
        """
        Yield tuples of nodes that belong to the same arithmetic group.

        Example:
            f = parse("y ~ x") + "z ~ b" + "u ~ v" + family()
            g = f + f

            list(g.iter_summands()) ->
            [
              (bf_yx, bf_zb, bf_uv, family),
              (bf_yx, bf_zb, bf_uv, family),
            ]
        """

        def _groups(node: Node) -> Iterator[list[FormulaPart | ProxyListSexpVector]]:
            # Leaf node: single bf/family/etc
            if isinstance(node, (FormulaPart, ProxyListSexpVector)):
                return ([node],)  # one group with one element

            if isinstance(node, list):
                # If any child is a list, this node represents a "+"
                # between sub-expressions, so recurse into each child.
                if any(isinstance(child, list) for child in node):
                    for child in node:
                        yield from _groups(child)
                else:
                    # All children are leaves -> one summand
                    out: list[FormulaPart | ProxyListSexpVector] = []
                    for child in node:
                        if isinstance(child, (FormulaPart, ProxyListSexpVector, Sexp)):
                            child = cast(FormulaPart | ProxyListSexpVector, child)
                            out.append(child)
                        else:
                            raise TypeError(
                                f"Unexpected leaf node type in FormulaConstruct: {type(child)!r}"
                            )
                    yield out
                return

            raise TypeError(f"Unexpected node type in FormulaConstruct: {type(node)!r}")

        # self._parts is always a list[Node]
        for group in _groups(self._parts):
            yield tuple(group)

    # Make __iter__ return summands by default
    def __iter__(self) -> Iterator[Summand]:
        return self.iter_summands()

    def iterate(self) -> Iterator[FormulaPart | ProxyListSexpVector]:
        """Yield FormulaPart objects in left-to-right addition order."""

        def _walk(node: Node) -> Iterator[FormulaPart | ProxyListSexpVector]:
            if isinstance(node, FormulaPart):
                yield node
            elif isinstance(node, ProxyListSexpVector):
                yield node
            elif isinstance(node, list):
                for child in node:
                    yield from _walk(child)
            else:
                raise TypeError(
                    f"Unexpected node type in FormulaConstruct: {type(node)!r}"
                )

        for root in self._parts:
            yield from _walk(root)

    def __str__(self) -> str:
        return self._pretty(self._parts)

    def _pretty(self, node, _outer=True) -> str:
        if isinstance(node, FormulaPart):
            return str(node)

        if isinstance(node, ProxyListSexpVector):
            return _sexp_to_str(node)

        if isinstance(node, list):
            # Pretty-print each child
            rendered = [self._pretty(child, _outer=False) for child in node]

            # If only one child, no parentheses needed
            if len(rendered) == 1:
                return rendered[0]

            # Multiple children → join with " + "
            inner = " + ".join(rendered)
            if _outer:
                return inner
            else:
                return f"({inner})"

        raise TypeError(f"Unexpected node type {type(node)!r} in pretty-printer")

    def __repr__(self) -> str:
        return self.__str__()
