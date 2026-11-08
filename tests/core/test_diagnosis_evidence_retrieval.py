"""Tests for the DiagnosisEvidenceRetrievalMIMIC3 task.

All patients are synthetic and built in memory, so the tests need no MIMIC data.
"""

import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import (
    DiagnosisEvidenceRetrievalMIMIC3,
    extract_likely_indicator_queries,
)

ER_TIME = datetime(2150, 3, 10, 8, 0)
REPORT_TEXT = (
    "FINDINGS: Crescentic extra-axial collection. "
    "IMPRESSION: Concerning for subdural hematoma along the left convexity."
)
ATTRS = {
    "admissions": ["hadm_id", "admission_type", "edregtime"],
    "noteevents": ["row_id", "hadm_id", "category", "description", "text"],
    "diagnoses_icd": ["hadm_id", "icd9_code"],
}


def _row(event_type: str, timestamp: datetime, **attrs: str) -> dict:
    """One event row in the wide format that Patient expects."""
    row = {"patient_id": "p1", "event_type": event_type, "timestamp": timestamp}
    for table, names in ATTRS.items():
        for name in names:
            row[f"{table}/{name}"] = attrs.get(name) if table == event_type else None
    return row


def _note(
    row_id: int,
    timestamp: datetime,
    category: str = "Nursing/other",
    description: str = "Report",
    text: str = "Stable overnight.",
) -> dict:
    return _row(
        "noteevents",
        timestamp,
        row_id=str(row_id),
        hadm_id="10",
        category=category,
        description=description,
        text=text,
    )


def _patient(rows: list) -> Patient:
    schema = {"patient_id": pl.Utf8, "event_type": pl.Utf8, "timestamp": pl.Datetime}
    schema.update({k: pl.Utf8 for k in rows[0] if k not in schema})
    return Patient(patient_id="p1", data_source=pl.DataFrame(rows, schema=schema))


def _er_patient(
    report_delay_hours: float = 3.0,
    num_prior_notes: int = 10,
    description: str = "CT HEAD W/O CONTRAST",
    admission_type: str = "EMERGENCY",
    edregtime: str | None = ER_TIME.isoformat(sep=" "),
    extra_rows: list | None = None,
) -> Patient:
    """ER patient with prior notes, a brain CT report and one later note."""
    report_time = ER_TIME + timedelta(hours=report_delay_hours)
    rows = [
        _row(
            "admissions",
            ER_TIME + timedelta(hours=1),
            hadm_id="10",
            admission_type=admission_type,
            edregtime=edregtime,
        ),
        _note(
            900,
            report_time,
            category="Radiology",
            description=description,
            text=REPORT_TEXT,
        ),
        _note(
            901,
            report_time + timedelta(days=2),
            category="Discharge summary",
            text="Discharged home. Subdural hematoma evacuated.",
        ),
    ]
    rows += [
        _note(100 + i, ER_TIME - timedelta(days=30 - i)) for i in range(num_prior_notes)
    ]
    return _patient(rows + (extra_rows or []))


class TestExtractLikelyIndicatorQueries(unittest.TestCase):
    def test_follows_paper_rules(self):
        text = (
            "Hypodensity concerning for infarct in the left MCA territory. "
            "Differential diagnosis includes glioma or metastasis. "
            "Changes likely represent old infarction."
        )
        self.assertEqual(
            extract_likely_indicator_queries(text),
            ["cerebral infarct", "glioma", "brain metastasis"],
        )

    def test_splits_conjunctions_and_deduplicates(self):
        text = (
            "Findings concerning for hemorrhage and edema. "
            "Diagnoses include hemorrhage, abscess."
        )
        self.assertEqual(
            extract_likely_indicator_queries(text),
            ["hemorrhage", "edema", "abscess"],
        )

    def test_drops_long_phrases_and_empty_text(self):
        long_text = "Concerning for one two three four five six seven eight."
        self.assertEqual(extract_likely_indicator_queries(long_text), [])
        self.assertEqual(
            extract_likely_indicator_queries(long_text, 10),
            ["one two three four five six seven eight"],
        )
        self.assertEqual(extract_likely_indicator_queries(""), [])
        self.assertEqual(extract_likely_indicator_queries(None), [])

    def test_keeps_existing_prefixes(self):
        text = "Likely represents cerebral infarction or brain metastasis."
        self.assertEqual(
            extract_likely_indicator_queries(text),
            ["cerebral infarction", "brain metastasis"],
        )


class TestDiagnosisEvidenceRetrievalMIMIC3(unittest.TestCase):
    def setUp(self):
        self.task = DiagnosisEvidenceRetrievalMIMIC3()

    def test_schema(self):
        self.assertEqual(self.task.task_name, "DiagnosisEvidenceRetrievalMIMIC3")
        self.assertEqual(self.task.input_schema, {"diagnosis": "raw", "notes": "raw"})
        self.assertEqual(self.task.output_schema, {"label": "binary"})

    def test_positive_sample(self):
        samples = self.task(_er_patient())
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["diagnosis"], "subdural hematoma")
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["hadm_id"], "10")
        self.assertEqual(sample["report_id"], "900")

    def test_notes_are_prior_only_and_ordered(self):
        notes = self.task(_er_patient())[0]["notes"]
        ids = [n["note_id"] for n in notes]
        self.assertEqual(ids, [str(100 + i) for i in range(10)])
        self.assertNotIn("900", ids)  # the report itself
        self.assertNotIn("901", ids)  # written after the report
        self.assertEqual(
            set(notes[0]), {"note_id", "category", "description", "charttime", "text"}
        )
        self.assertEqual(notes[0]["text"], "Stable overnight.")

    def test_min_notes(self):
        self.assertEqual(self.task(_er_patient(num_prior_notes=9)), [])
        task = DiagnosisEvidenceRetrievalMIMIC3(min_notes=9)
        self.assertEqual(len(task(_er_patient(num_prior_notes=9))), 1)

    def test_imaging_window(self):
        self.assertEqual(self.task(_er_patient(report_delay_hours=49)), [])
        task = DiagnosisEvidenceRetrievalMIMIC3(window_hours=72)
        self.assertEqual(len(task(_er_patient(report_delay_hours=49))), 1)

    def test_report_before_visit_is_ignored(self):
        self.assertEqual(self.task(_er_patient(report_delay_hours=-1)), [])

    def test_non_brain_imaging_is_ignored(self):
        patient = _er_patient(description="CT CHEST W/CONTRAST")
        self.assertEqual(self.task(patient), [])

    def test_visit_type(self):
        elective = _er_patient(admission_type="ELECTIVE", edregtime=None)
        self.assertEqual(self.task(elective), [])
        # Urgent admissions without an ER time use admittime (1 h after ER_TIME).
        urgent = _er_patient(admission_type="URGENT", edregtime=None)
        self.assertEqual(len(self.task(urgent)), 1)

    def test_negative_queries(self):
        diagnoses = [
            _row(
                "diagnoses_icd",
                ER_TIME + timedelta(days=3),
                hadm_id="10",
                icd9_code="4321",
            ),
        ]
        pool = {
            "4321": "Subdural hemorrhage",  # patient's own code
            "4320": "Nontraumatic extradural hemorrhage",  # same category
            "0389": "Septicemia NOS",
            "5770": "Acute pancreatitis",
        }
        task = DiagnosisEvidenceRetrievalMIMIC3(negative_pool=pool, num_negatives=5)
        samples = task(_er_patient(extra_rows=diagnoses))
        negatives = sorted(s["diagnosis"] for s in samples if s["label"] == 0)
        self.assertEqual(negatives, ["acute pancreatitis", "septicemia nos"])
        self.assertEqual(sum(s["label"] for s in samples), 1)

    def test_negative_queries_are_deterministic(self):
        pool = {f"{code:04d}": f"Diagnosis {code}" for code in range(1000, 1050)}
        task = DiagnosisEvidenceRetrievalMIMIC3(negative_pool=pool, num_negatives=2)
        first = [s["diagnosis"] for s in task(_er_patient())]
        second = [s["diagnosis"] for s in task(_er_patient())]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_note_categories(self):
        rows = [
            _note(
                200 + i, ER_TIME - timedelta(days=5 - i), category="Discharge summary"
            )
            for i in range(2)
        ]
        task = DiagnosisEvidenceRetrievalMIMIC3(
            note_categories=["discharge summary"], min_notes=2
        )
        notes = task(_er_patient(extra_rows=rows))[0]["notes"]
        self.assertEqual([n["note_id"] for n in notes], ["200", "201"])

    def test_report_without_queries(self):
        patient = _patient(
            [
                _row(
                    "admissions",
                    ER_TIME,
                    hadm_id="10",
                    admission_type="EMERGENCY",
                    edregtime=ER_TIME.isoformat(sep=" "),
                ),
                _note(
                    900,
                    ER_TIME + timedelta(hours=2),
                    category="Radiology",
                    description="MR HEAD",
                    text="No acute intracranial process.",
                ),
            ]
            + [_note(100 + i, ER_TIME - timedelta(days=i + 1)) for i in range(10)]
        )
        self.assertEqual(self.task(patient), [])

    def test_patient_without_notes(self):
        patient = _patient(
            [
                _row(
                    "admissions",
                    ER_TIME,
                    hadm_id="10",
                    admission_type="EMERGENCY",
                    edregtime=ER_TIME.isoformat(sep=" "),
                ),
            ]
        )
        self.assertEqual(self.task(patient), [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DiagnosisEvidenceRetrievalMIMIC3(window_hours=0)
        with self.assertRaises(ValueError):
            DiagnosisEvidenceRetrievalMIMIC3(min_notes=-1)

    def test_pre_filter_keeps_patients_with_notes(self):
        df = pl.LazyFrame(
            {
                "patient_id": ["p1", "p1", "p2"],
                "event_type": ["admissions", "noteevents", "admissions"],
            }
        )
        kept = self.task.pre_filter(df).collect()
        self.assertEqual(kept["patient_id"].to_list(), ["p1", "p1"])

    def test_samples_load_into_sample_dataset(self):
        pool = {"0389": "Septicemia NOS"}
        task = DiagnosisEvidenceRetrievalMIMIC3(negative_pool=pool)
        samples = task(_er_patient())
        dataset = create_sample_dataset(samples, task.input_schema, task.output_schema)
        self.assertEqual(len(dataset), 2)
        self.assertIsInstance(dataset[0]["notes"], list)
        self.assertEqual(dataset[0]["notes"][0]["note_id"], "100")
        self.assertEqual(
            sorted(float(dataset[i]["label"]) for i in range(2)), [0.0, 1.0]
        )


if __name__ == "__main__":
    unittest.main()
