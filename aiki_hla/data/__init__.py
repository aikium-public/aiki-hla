"""Data pipeline for AIKI-HLA.

Public surface:
    run_audit_gate     — six-mode contamination check; refuses to proceed
                         if any of the six failure modes named in the
                         manuscript Methods §"Data sources and contamination
                         prevention" is detected.

The audit gate is the canonical defense the manuscript names against
contamination at training entry. It is the same gate that runs in the
released training pipeline.
"""
from aiki_hla.data.audit_gate import run_audit_gate, AuditReport, AuditFailure  # noqa: F401

__all__ = ["run_audit_gate", "AuditReport", "AuditFailure"]
