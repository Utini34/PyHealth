"""Diagnosis evidence retrieval task for MIMIC-III.

Authors: Roland Langer (rlanger3@gatech.edu), Ripan Nandi (rnandi30@gatech.edu)
Paper: Ahsan et al., "Retrieving Evidence from EHRs with LLMs: Possibilities and
    Challenges", CHIL 2024. https://arxiv.org/abs/2309.04550
Description: Builds patient-diagnosis instances for zero-shot evidence retrieval.
    Each instance pairs a brain imaging report, taken within 48 hours of an
    emergency or urgent-care visit, with a query diagnosis and the notes written
    before the report. Query diagnoses come from "likely indicator" phrases in the
    report (label 1) or are ICD-9 diagnoses the patient never had (label 0).
"""

import random
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from pyhealth.data import Event, Patient

from .base_task import BaseTask

BRAIN_IMAGING_PATTERN = r"\b(?:CT|CTA|MR|MRI|MRA)\b.*\b(?:HEAD|BRAIN)\b"
"""Matches MIMIC-III radiology descriptions such as ``CT HEAD W/O CONTRAST``."""

_INDICATOR = re.compile(
    r"\b(?:likely represents?|concerning for|diagnos[ie]s includes?)\b"
    r"(?P<phrase>[^.;:\n?]*)",
    re.IGNORECASE,
)
_SPLIT = re.compile(r",|/|\bor\b|\band\b|\bversus\b|\bvs\b", re.IGNORECASE)
_CUT = re.compile(
    r"\b(?:in|within|involving|at|along|from|on|of the|which|that|given|"
    r"with|without|is|are|was|were|may|cannot|not)\b.*$",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^(?:(?:a|an|the|either|possible|probable)\s+)+")
_INFARCT = re.compile(r"(?<!cerebral )(?<!myocardial )\b(infarcts?|infarctions?)\b")
_METASTASIS = re.compile(r"(?<!brain )\b(metastas[ie]s|metastatic disease)\b")


def extract_likely_indicator_queries(report_text: str, max_words: int = 6) -> list[str]:
    """Extracts query diagnoses that follow likely-indicator phrases.

    Follows Appendix D of Ahsan et al. (2024): conditions after "likely
    represent", "concerning for" and "diagnosis include" become queries,
    "old" findings such as "old infarction" are skipped, and "cerebral" or
    "brain" is added in front of "infarction" and "metastasis". Each phrase is
    cut at the end of its clause, split into alternatives ("X or Y"), and
    trimmed before location words such as "in the left frontal lobe".

    Args:
        report_text: Free text of a radiology report.
        max_words: Longest query kept, in words. Longer phrases are usually
            descriptions rather than diagnoses.

    Returns:
        Lower-cased, de-duplicated queries in order of appearance.

    Examples:
        >>> extract_likely_indicator_queries(
        ...     "Hypodensity concerning for infarct in the left MCA territory. "
        ...     "Differential diagnosis includes glioma or metastasis."
        ... )
        ['cerebral infarct', 'glioma', 'brain metastasis']
    """
    queries: list[str] = []
    for match in _INDICATOR.finditer(report_text or ""):
        for part in _SPLIT.split(match.group("phrase")):
            query = _CUT.sub("", part.lower())
            query = re.sub(r"[^a-z0-9\- ]", " ", query)
            query = _LEADING.sub("", " ".join(query.split()))
            if not query or query.startswith("old ") or len(query.split()) > max_words:
                continue
            query = _INFARCT.sub(r"cerebral \1", query)
            query = _METASTASIS.sub(r"brain \1", query)
            if query not in queries:
                queries.append(query)
    return queries


def _to_datetime(value: Any) -> datetime | None:
    """Parses a MIMIC timestamp that may be a string, datetime or missing."""
    if isinstance(value, datetime):
        return value
    if value is None or str(value).strip() in ("", "None", "nan", "<NA>"):
        return None
    return datetime.fromisoformat(str(value))


class DiagnosisEvidenceRetrievalMIMIC3(BaseTask):
    """Patient-diagnosis instances for zero-shot EHR evidence retrieval.

    Reproduces the MIMIC-III cohort of Ahsan et al. (2024). For every admission
    with an emergency or urgent-care visit, the first brain imaging report
    within ``window_hours`` of the visit is the anchor. The patient's notes
    written before that report (across all admissions) are the evidence
    source, and the admission is kept only if there are at least
    ``min_notes`` of them. One sample is made per query diagnosis:

    * label 1: a condition that follows a likely-indicator phrase in the report
      (see :func:`extract_likely_indicator_queries`);
    * label 0: an ICD-9 diagnosis the patient never had, drawn from
      ``negative_pool``. Codes that share a 3-character category with any of
      the patient's codes are also excluded. Used to measure false positives
      (Appendix A.1 of the paper).

    The emergency visit time is ``edregtime``; urgent and emergency admissions
    without it fall back to ``admittime``. Notes stay as raw text so that an
    LLM retriever (e.g. :class:`~pyhealth.models.LLMEvidenceRetriever`) can
    split them into prompt-sized chunks with its own tokenizer.

    Args:
        window_hours: Maximum hours between the visit and the imaging report.
        min_notes: Minimum number of notes written before the report.
        imaging_pattern: Regex (case-insensitive) on the note ``description``
            that marks a radiology note as brain imaging.
        note_categories: If given, only notes in these categories (e.g.
            ``["Discharge summary", "Radiology"]``) are used as evidence and
            counted towards ``min_notes``.
        negative_pool: Mapping from ICD-9 diagnosis code to its name, e.g.
            loaded from ``D_ICD_DIAGNOSES.csv``. If ``None``, no label-0
            samples are made. Requires the ``diagnoses_icd`` table.
        num_negatives: Number of label-0 queries per anchor report.
        seed: Seed for drawing negative queries. The draw depends only on the
            seed, patient and report, so it is the same on every run.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import DiagnosisEvidenceRetrievalMIMIC3
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["noteevents", "diagnoses_icd"],
        ... )
        >>> task = DiagnosisEvidenceRetrievalMIMIC3(window_hours=48, min_notes=10)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["diagnosis"], samples[0]["label"]  # doctest: +SKIP
        ('subdural hematoma', 1)
    """

    task_name: str = "DiagnosisEvidenceRetrievalMIMIC3"
    input_schema: ClassVar[dict[str, str]] = {"diagnosis": "raw", "notes": "raw"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        window_hours: float = 48.0,
        min_notes: int = 10,
        imaging_pattern: str = BRAIN_IMAGING_PATTERN,
        note_categories: Sequence[str] | None = None,
        negative_pool: Mapping[str, str] | None = None,
        num_negatives: int = 1,
        seed: int = 42,
    ) -> None:
        """Initializes the task. See the class docstring for the arguments."""
        super().__init__()
        if window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if min_notes < 0 or num_negatives < 0:
            raise ValueError("min_notes and num_negatives must be non-negative")
        self.window = timedelta(hours=window_hours)
        self.min_notes = min_notes
        self.imaging_regex = re.compile(imaging_pattern, re.IGNORECASE)
        self.note_categories = (
            {c.strip().lower() for c in note_categories}
            if note_categories is not None
            else None
        )
        self.negative_pool = dict(negative_pool or {})
        self.num_negatives = num_negatives
        self.seed = seed

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps only patients that have at least one note.

        Args:
            df: Global event dataframe of the dataset.

        Returns:
            The dataframe restricted to patients with ``noteevents`` rows.
        """
        with_notes = df.filter(pl.col("event_type") == "noteevents").select(
            "patient_id"
        )
        return df.join(with_notes.unique(), on="patient_id", how="semi")

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds the evidence retrieval samples of one patient.

        Args:
            patient: Patient with ``admissions`` and ``noteevents`` events, and
                ``diagnoses_icd`` events if ``negative_pool`` is set.

        Returns:
            One dict per (anchor report, query diagnosis) with keys
            ``patient_id``, ``hadm_id``, ``report_id``, ``diagnosis`` (query
            text), ``notes`` (list of dicts with ``note_id``, ``category``,
            ``description``, ``charttime`` and ``text``, oldest first) and
            ``label`` (1 for likely-indicator queries, 0 for negatives).
        """
        notes = [
            n
            for n in patient.get_events(event_type="noteevents")
            if n.timestamp is not None
        ]
        patient_codes = {
            str(e.icd9_code)
            for e in patient.get_events(event_type="diagnoses_icd")
            if e.icd9_code is not None
        }
        samples: list[dict[str, Any]] = []
        for admission in patient.get_events(event_type="admissions"):
            visit = self._visit_time(admission)
            if visit is None:
                continue
            report = self._first_brain_imaging(notes, visit)
            if report is None:
                continue
            prior = self._prior_notes(notes, report)
            if len(prior) < self.min_notes:
                continue
            queries = [(q, 1) for q in extract_likely_indicator_queries(report.text)]
            queries += [
                (q, 0) for q in self._negative_queries(patient, report, patient_codes)
            ]
            for diagnosis, label in queries:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "hadm_id": str(admission.hadm_id),
                        "report_id": str(report.row_id),
                        "diagnosis": diagnosis,
                        "notes": prior,
                        "label": label,
                    }
                )
        return samples

    def _visit_time(self, admission: Event) -> datetime | None:
        """Returns the ER registration time, or admittime for urgent visits."""
        edregtime = _to_datetime(admission.attr_dict.get("edregtime"))
        if edregtime is not None:
            return edregtime
        admission_type = str(admission.attr_dict.get("admission_type", "")).upper()
        if admission_type in ("EMERGENCY", "URGENT"):
            return admission.timestamp
        return None

    def _first_brain_imaging(self, notes: list[Event], visit: datetime) -> Event | None:
        """Returns the earliest brain imaging report within the window."""
        for note in notes:
            if not visit <= note.timestamp <= visit + self.window:
                continue
            category = str(note.attr_dict.get("category") or "").strip().lower()
            description = str(note.attr_dict.get("description") or "")
            if (
                category == "radiology"
                and self.imaging_regex.search(description)
                and note.attr_dict.get("text")
            ):
                return note
        return None

    def _prior_notes(self, notes: list[Event], report: Event) -> list[dict[str, str]]:
        """Returns the notes written before ``report``, oldest first."""
        prior = []
        for note in notes:
            if note.timestamp >= report.timestamp or not note.attr_dict.get("text"):
                continue
            category = str(note.attr_dict.get("category") or "").strip()
            if (
                self.note_categories is not None
                and category.lower() not in self.note_categories
            ):
                continue
            prior.append(
                {
                    "note_id": str(note.attr_dict.get("row_id")),
                    "category": category,
                    "description": str(note.attr_dict.get("description") or ""),
                    "charttime": note.timestamp.isoformat(),
                    "text": str(note.attr_dict["text"]),
                }
            )
        return prior

    def _negative_queries(
        self, patient: Patient, report: Event, patient_codes: set[str]
    ) -> list[str]:
        """Draws ICD-9 diagnosis names the patient never had."""
        if not self.negative_pool or self.num_negatives == 0:
            return []
        categories = {code[:3] for code in patient_codes}
        candidates = sorted(
            code
            for code in self.negative_pool
            if code not in patient_codes and code[:3] not in categories
        )
        rng = random.Random(f"{self.seed}:{patient.patient_id}:{report.row_id}")
        chosen = rng.sample(candidates, min(self.num_negatives, len(candidates)))
        return [self.negative_pool[code].strip().lower() for code in chosen]
