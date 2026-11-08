pyhealth.tasks.DiagnosisEvidenceRetrievalMIMIC3
===============================================

Patient-diagnosis instances for zero-shot evidence retrieval from clinical notes,
following the MIMIC-III setup of `Ahsan et al. (2024)
<https://arxiv.org/abs/2309.04550>`_. Each sample pairs a brain imaging report taken
within 48 hours of an emergency or urgent-care visit with a query diagnosis and the
patient's earlier notes.

.. autoclass:: pyhealth.tasks.diagnosis_evidence_retrieval.DiagnosisEvidenceRetrievalMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.diagnosis_evidence_retrieval.extract_likely_indicator_queries
