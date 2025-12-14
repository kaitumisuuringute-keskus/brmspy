from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from importlinter import Contract, ContractCheck, fields, output


@dataclass(frozen=True)
class Violation:
    importer: str
    line_number: int
    line_contents: str


def _match_any(module: str, patterns: set[str]) -> bool:
    for pat in patterns:
        # treat ** as "anything"
        pat = pat.replace("**", "*")
        if fnmatch.fnmatch(module, pat):
            return True
    return False


class ForbidExternalSubmoduleImportContract(Contract):
    """
    Forbid importing a specific external submodule (e.g. rpy2.robjects)
    from most places, with an allowlist.

    Works around Import Linter's "external subpackages are invalid" rule by
    inspecting import lines instead of treating the external subpackage as a node.
    """

    source_modules = fields.SetField(subfield=fields.StringField(), default={"**"})
    forbidden_submodule = fields.StringField()  # e.g. rpy2.robjects
    allowed_importers = fields.SetField(subfield=fields.StringField())

    def check(self, graph, verbose: bool) -> ContractCheck:
        forbidden = self.forbidden_submodule.strip()
        top_level = forbidden.split(".", 1)[0]  # "rpy2"
        # First segment after top-level, for catching: "from rpy2 import robjects"
        sub_first = forbidden.split(".", 2)[1] if "." in forbidden else ""

        from_top_import_re = re.compile(
            rf"^\s*from\s+{re.escape(top_level)}\s+import\s+.*\b{re.escape(sub_first)}\b",
        )

        violations: list[Violation] = []

        modules = getattr(graph, "modules", None)
        if modules is None:
            # fallback if grimp stores nodes differently
            g = getattr(graph, "graph", None)
            modules = list(getattr(g, "nodes", [])) if g is not None else []

        for importer in modules:
            if not _match_any(importer, self.source_modules):
                continue
            if _match_any(importer, self.allowed_importers):
                continue

            # We ask the graph for imports to the TOP-LEVEL external package,
            # then filter down by the actual import line contents.
            try:
                details = graph.get_import_details(
                    importer=importer, imported=top_level
                )
            except Exception:
                continue

            for d in details:
                line = (d.get("line_contents") or "").strip()
                if not line:
                    continue

                # Direct submodule imports
                if forbidden in line:
                    violations.append(
                        Violation(importer, int(d.get("line_number") or 0), line)
                    )
                    continue

                # Also catch: from rpy2 import robjects
                if sub_first and from_top_import_re.match(line):
                    violations.append(
                        Violation(importer, int(d.get("line_number") or 0), line)
                    )

        return ContractCheck(
            kept=(len(violations) == 0),
            metadata={"violations": violations, "forbidden_submodule": forbidden},
        )

    def render_broken_contract(self, check: ContractCheck) -> None:
        forbidden = check.metadata.get("forbidden_submodule", "<unknown>")
        violations: list[Violation] = check.metadata.get("violations", [])

        output.print_error(
            f"Forbidden import of {forbidden} outside allowed importers:",
            bold=True,
        )
        output.new_line()

        for v in violations:
            output.indent_cursor()
            output.print_error(f"{v.importer}:{v.line_number}: {v.line_contents}")
