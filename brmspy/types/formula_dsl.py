"""
Formula DSL types.

This module contains lightweight Python types used by the public formula helpers
([`bf()`][brmspy.brms.bf], [`set_rescor()`][brmspy.brms.set_rescor], etc.)
to represent brms formula expressions in a structured way.

The main entry point for end users is the set of helpers exposed via
[`brmspy.brms`][brmspy.brms]. Those helpers return `FormulaConstruct` instances
which can be combined with `+` to build multivariate or compound models.

Notes
-----
- These objects are *data containers*; the execution (turning them into actual R
  formula objects) happens in the worker process.
- `ProxyListSexpVector` values inside the tree represent R-side objects (for
  example brms family objects) and are only meaningful while the worker process
  that created them is alive.

Examples
--------
Compose formula parts using the public helpers:

```python
from brmspy.brms import bf, set_rescor

f = bf("y ~ x") + bf("z ~ 1") + set_rescor(True)
print(f)
```
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, Union, cast, get_args

from rpy2.rinterface_lib.sexp import Sexp

from .brms_results import ProxyListSexpVector


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
    """
    A single formula helper invocation.

    Instances of this type represent a call like `bf("y ~ x")` or `set_rescor(True)`
    without executing anything. They are primarily used as nodes inside a
    [`FormulaConstruct`][brmspy.types.formula_dsl.FormulaConstruct].

    Parameters
    ----------
    _fun : Literal[...]
        Whitelisted formula helper name.
    _args : Sequence[Primitive]
        Positional arguments for the helper.
    _kwargs : Mapping[str, Primitive]
        Keyword arguments for the helper.

    Notes
    -----
    This is a low-level type. Most users should construct these via the public
    helper functions in [`brmspy.brms`][brmspy.brms].
    """

    _fun: _FORMULA_FUNCTION_WHITELIST
    _args: Sequence[Primitive]
    _kwargs: Mapping[str, Primitive]

    def __post_init__(self):
        """Validate `_fun`, `_args`, and `_kwargs` types after construction."""
        # Validate function name first
        if self._fun not in get_args(_FORMULA_FUNCTION_WHITELIST):
            raise ValueError(
                f"FormulaPart._fun must be one of {_FORMULA_FUNCTION_WHITELIST!r}, "
                f"got {self._fun!r}"
            )

        # Enforce _args is a list
        if not isinstance(self._args, Sequence):
            raise TypeError(
                f"FormulaPart._args must be a Sequence, got {type(self._args).__name__}"
            )

        # Enforce _kwargs is a dict
        if not isinstance(self._kwargs, Mapping):
            raise TypeError(
                f"FormulaPart._kwargs must be a Mapping, got {type(self._kwargs).__name__}"
            )

    def __str__(self) -> str:
        """Render a readable `fun(arg1, ..., kw=...)` representation."""
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
    """
    Best-effort pretty-printer for selected R objects.

    Currently this extracts `family()` / `link` information from the printed
    representation of a brms family object.

    Parameters
    ----------
    o : ProxyListSexpVector
        R object wrapper returned by the worker.

    Returns
    -------
    str
        Readable string if recognized, otherwise an empty string.
    """
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
    """
    A composite formula expression built from parts.

    `FormulaConstruct` stores a tree of nodes (`FormulaPart` and/or R objects)
    representing expressions combined with `+`. It is primarily created by
    calling the public formula helpers exposed by [`brmspy.brms`][brmspy.brms].

    Notes
    -----
    The `+` operator supports grouping:

    - `a + b + c` becomes a single summand (one “group”)
    - `(a + b) + (a + b)` becomes two summands (two “groups”)

    Use [`iter_summands()`][brmspy.types.formula_dsl.FormulaConstruct.iter_summands]
    to iterate over these groups in a deterministic way.
    """

    _parts: list[Node]

    @classmethod
    def _formula_parse(cls, obj: Other) -> "FormulaConstruct":
        """
        Convert a supported value into a `FormulaConstruct`.

        Parameters
        ----------
        obj
            One of: `FormulaConstruct`, `FormulaPart`, string (interpreted as `bf(<string>)`),
            or `ProxyListSexpVector`.

        Returns
        -------
        FormulaConstruct
        """
        if isinstance(obj, FormulaConstruct):
            return obj
        if isinstance(obj, ProxyListSexpVector):
            return FormulaConstruct(_parts=[obj])
        if isinstance(obj, FormulaPart):
            return FormulaConstruct(_parts=[obj])
        if isinstance(obj, str):
            part = FormulaPart(_fun="bf", _args=[obj], _kwargs={})
            return FormulaConstruct(_parts=[part])
        raise TypeError(
            f"Cannot parse object of type {type(obj)!r} into FormulaConstruct"
        )

    def __add__(self, other: Other):
        """
        Combine two formula expressions with `+`.

        Parameters
        ----------
        other
            Value to add. Strings are treated as `bf(<string>)`.

        Returns
        -------
        FormulaConstruct
            New combined expression.
        """
        if isinstance(other, (FormulaPart, str, ProxyListSexpVector)):
            other = FormulaConstruct._formula_parse(other)

        if not isinstance(other, FormulaConstruct):
            raise ArithmeticError(
                "When adding values to formula, they must be FormulaConstruct or parseable to FormulaConstruct"
            )

        if len(other._parts) <= 1:
            return FormulaConstruct(_parts=self._parts + other._parts)
        else:
            return FormulaConstruct(_parts=[self._parts, other._parts])

    def __radd__(self, other: Other) -> "FormulaConstruct":
        """Support `"y ~ x" + bf("z ~ 1")` by coercing the left operand."""
        return self._formula_parse(other) + self

    def iter_summands(self) -> Iterator[Summand]:
        """
        Iterate over arithmetic groups (summands).

        Returns
        -------
        Iterator[tuple[FormulaPart | ProxyListSexpVector, ...]]
            Each yielded tuple represents one summand/group.

        Examples
        --------
        ```python
        from brmspy.brms import bf, gaussian, set_rescor

        f = bf("y ~ x") + gaussian() + set_rescor(True)
        for summand in f.iter_summands():
            print(summand)
        ```
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
        """Alias for [`iter_summands()`][brmspy.types.formula_dsl.FormulaConstruct.iter_summands]."""
        return self.iter_summands()

    def iterate(self) -> Iterator[FormulaPart | ProxyListSexpVector]:
        """
        Iterate over all leaf nodes in left-to-right order.

        This flattens the expression tree, unlike
        [`iter_summands()`][brmspy.types.formula_dsl.FormulaConstruct.iter_summands], which
        respects grouping.

        Returns
        -------
        Iterator[FormulaPart | ProxyListSexpVector]
        """

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

        if isinstance(node, (ProxyListSexpVector, Sexp)):
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
