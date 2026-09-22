"""Data-space-driven NMF objective resolution.

The RecoVar NMF (selection + evaluation) historically pinned a single kwarg set
for ``nico2_lib.NmfPredictor`` (``solver="cd"``, ``beta_loss="frobenius"``,
``init="nndsvd"``, ``max_iter=1000``) regardless of what matrix was fed in.

This helper ties the factorization objective to the *count-input choice* already
made by the user:

* ``raw`` integer counts  -> ``mu`` solver + Kullback-Leibler ``beta_loss``
  (the Poisson-appropriate objective for UMI counts). ``init="nndsvda"`` is
  mandatory for ``mu`` (it cannot escape the exact zeros ``nndsvd`` produces),
  and ``mu`` needs more iterations than ``cd``.
* ``lognorm`` (log1p-normalised) counts -> ``cd`` solver + Frobenius
  ``beta_loss`` + ``nndsvd`` + ``max_iter=1000`` — **byte-identical** to the old
  pinned kwarg set, so the log-normalised path is unchanged.

An explicit ``override`` (CLI ``--nmf_objective``) can force either objective:

* ``"auto"``      -> derive from ``counts_input`` (default)
* ``"frobenius"`` -> ``cd`` / Frobenius / ``nndsvd`` / 1000   (any input)
* ``"kl"``        -> ``mu`` / Kullback-Leibler / ``nndsvda`` / 2000 (any input)

The returned dict is spread straight into ``NmfPredictor(...)`` alongside
``embedding_size`` / ``random_state`` (and the still-inert ``alpha_*`` /
``l1_ratio`` no-ops, if the caller keeps passing them).

Ported from the ``Code/RecoVar_mukl`` isolated cut into production
``Code/RecoVar`` (see ``docs/doc-pipeline/mukl-test_nmf-objective-experiment.md``
and ``docs/doc-pipeline/audit_3.md``) after that cut's own validation sweep
completed with no selection/evaluation failures and zero ``mu`` solver
``ConvergenceWarning``s.
"""

from __future__ import annotations

# Objective presets. Keep these two dicts the single source of truth.
_FROBENIUS = dict(solver="cd", beta_loss="frobenius", init="nndsvd", max_iter=1000)
_KL = dict(solver="mu", beta_loss="kullback-leibler", init="nndsvda", max_iter=2000)

NMF_OBJECTIVE_CHOICES = ("auto", "frobenius", "kl")


def resolve_nmf_objective(counts_input: str, override: str = "auto") -> dict:
    """Return the ``NmfPredictor`` solver/loss kwargs for a given count input.

    Args:
        counts_input: ``"raw"`` or ``"lognorm"`` — the matrix that will be fed to
            NMF (RecoVar's ``--dimred_counts_input`` / ``--nmf_counts_input``).
        override: ``"auto"`` (derive from ``counts_input``), ``"frobenius"``
            (force ``cd`` / Frobenius), or ``"kl"`` (force ``mu`` /
            Kullback-Leibler).

    Returns:
        ``dict`` with ``solver``, ``beta_loss``, ``init``, ``max_iter`` keys.

    Raises:
        ValueError: unknown ``counts_input`` or ``override``.
    """
    if override not in NMF_OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unknown nmf_objective override={override!r}. "
            f"Choose one of {NMF_OBJECTIVE_CHOICES}."
        )

    if override == "frobenius":
        return dict(_FROBENIUS)
    if override == "kl":
        return dict(_KL)

    # override == "auto" -> derive from the count-input choice
    if counts_input == "raw":
        return dict(_KL)
    if counts_input == "lognorm":
        return dict(_FROBENIUS)
    raise ValueError(
        f"Unknown counts_input={counts_input!r}. Choose 'raw' or 'lognorm' "
        f"(or pass an explicit nmf_objective override)."
    )


def describe_nmf_objective(counts_input: str, override: str = "auto") -> str:
    """One-line human-readable summary for logs / parameter JSON."""
    kw = resolve_nmf_objective(counts_input, override)
    return (
        f"nmf_objective={override} (counts_input={counts_input}) -> "
        f"solver={kw['solver']}, beta_loss={kw['beta_loss']}, "
        f"init={kw['init']}, max_iter={kw['max_iter']}"
    )
