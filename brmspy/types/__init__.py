"""
Public type definitions for brmspy.

This package collects the stable Python types that appear in the public API and
in return values from [`brmspy.brms`][brmspy.brms].

Highlights
----------
- Result containers returned by brms wrapper calls: see
  [`brmspy.types.brms_results`][brmspy.types.brms_results].
- Formula DSL node types used by the formula helpers:
  [`brmspy.types.formula_dsl`][brmspy.types.formula_dsl].
- Runtime/environment/session configuration structures:
  [`brmspy.types.runtime`][brmspy.types.runtime] and
  [`brmspy.types.session`][brmspy.types.session].
- Shared-memory helper types used by the IPC codecs:
  [`brmspy.types.shm`][brmspy.types.shm] and
  [`brmspy.types.shm_extensions`][brmspy.types.shm_extensions].

Notes
-----
brmspy runs embedded R in an isolated worker process. Any R objects that appear
in Python results are represented as lightweight handles (see
[`SexpWrapper`][brmspy.types.session.SexpWrapper]) and can only be used by passing
them back into brmspy calls while the same worker process is alive.
"""
