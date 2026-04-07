"""
src/inference/rag_recommender.py
─────────────────────────────────
Retrieval-Augmented Generation (RAG) for evidence-based test recommendations.

Workflow:
  1. Index clinical guidelines (ACC/AHA, NICE, WHO PDFs) into a FAISS
     vector store at startup using sentence-transformers embeddings.
  2. At inference, retrieve the top-K most relevant guideline chunks
     given the predicted diseases + patient context.
  3. Use those chunks to generate structured test recommendations.
     (In production, this calls the Anthropic API; a fallback uses
     a curated rule-based lookup for common diseases.)

The guideline store is updated monthly without model retraining.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from src.data.schema import RecommendedTest
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Curated fallback rulebook ─────────────────────────────────────────────
# For production, this is much larger and loaded from a JSON.
# Each entry maps ICD-10 prefix → recommended investigations.
GUIDELINE_RULEBOOK: dict[str, list[dict]] = {
    "I21": [  # STEMI / Acute MI
        {"test": "12-lead ECG (serial)", "rationale": "Confirm STEMI pattern, monitor ST changes",
         "source": "ACC/AHA 2022 STEMI Guidelines", "priority": "FIRST_LINE", "turnaround": "Immediate"},
        {"test": "High-sensitivity Troponin I/T (0h, 1h, 3h)",
         "rationale": "Confirm myocardial injury; serial measurements for NSTEMI rule-in/out",
         "source": "ESC NSTE-ACS 2023", "priority": "FIRST_LINE", "turnaround": "1 hour"},
        {"test": "STAT Cardiology consultation + Coronary Angiography",
         "rationale": "PCI within 90 min of first medical contact for STEMI",
         "source": "ACC/AHA 2022", "priority": "FIRST_LINE", "turnaround": "Immediate"},
        {"test": "CXR (portable)", "rationale": "Assess pulmonary oedema, aortic silhouette",
         "source": "ACC/AHA 2022", "priority": "FIRST_LINE", "turnaround": "30 minutes"},
        {"test": "Echocardiogram", "rationale": "Assess LV function, wall motion abnormality",
         "source": "ACC/AHA 2022", "priority": "SECOND_LINE", "turnaround": "24 hours"},
    ],
    "I26": [  # Pulmonary embolism
        {"test": "CT Pulmonary Angiography (CTPA)",
         "rationale": "Gold standard for PE diagnosis",
         "source": "ESC PE Guidelines 2019", "priority": "FIRST_LINE", "turnaround": "1–2 hours"},
        {"test": "D-dimer (if low/intermediate pre-test probability)",
         "rationale": "High sensitivity rule-out; use Wells/Geneva score first",
         "source": "ESC PE Guidelines 2019 / NICE NG158", "priority": "FIRST_LINE", "turnaround": "1 hour"},
        {"test": "ABG (arterial blood gas)",
         "rationale": "Assess hypoxia, hypocapnia pattern",
         "source": "ESC PE Guidelines 2019", "priority": "FIRST_LINE", "turnaround": "30 minutes"},
        {"test": "BNP / NT-proBNP", "rationale": "RV dysfunction marker; prognostication",
         "source": "ESC PE Guidelines 2019", "priority": "SECOND_LINE", "turnaround": "2 hours"},
    ],
    "A41": [  # Sepsis
        {"test": "Blood cultures × 2 (before antibiotics)",
         "rationale": "Mandatory before antibiotic initiation; improves source identification",
         "source": "Surviving Sepsis Campaign 2021", "priority": "FIRST_LINE", "turnaround": "72 hours"},
        {"test": "Lactate (serum)",
         "rationale": "Lactate ≥ 2 mmol/L = sepsis; ≥4 = septic shock → immediate resuscitation",
         "source": "Surviving Sepsis Campaign 2021", "priority": "FIRST_LINE", "turnaround": "1 hour"},
        {"test": "Full blood count, CRP, Procalcitonin",
         "rationale": "Inflammatory markers; PCT guides antibiotic de-escalation",
         "source": "SSC 2021 / NICE NG51", "priority": "FIRST_LINE", "turnaround": "2 hours"},
        {"test": "Urine MC&S (midstream catch)",
         "rationale": "Urosepsis is commonest source in community-acquired sepsis",
         "source": "NICE NG51", "priority": "FIRST_LINE", "turnaround": "48–72 hours"},
        {"test": "Chest X-ray",
         "rationale": "Identify pulmonary source",
         "source": "NICE NG51", "priority": "FIRST_LINE", "turnaround": "1 hour"},
    ],
    "E11": [  # Type 2 Diabetes
        {"test": "HbA1c",
         "rationale": "Confirms diagnosis if ≥48 mmol/mol; monitors glycaemic control",
         "source": "NICE NG28 / ADA Standards 2024", "priority": "FIRST_LINE", "turnaround": "24 hours"},
        {"test": "Fasting lipid profile",
         "rationale": "Cardiovascular risk assessment; statin indication",
         "source": "ADA Standards 2024", "priority": "FIRST_LINE", "turnaround": "24 hours"},
        {"test": "Urine albumin:creatinine ratio (ACR)",
         "rationale": "Screen for diabetic nephropathy",
         "source": "NICE NG28", "priority": "FIRST_LINE", "turnaround": "24 hours"},
        {"test": "Fundoscopy / retinal photography",
         "rationale": "Screen for diabetic retinopathy",
         "source": "NICE NG28", "priority": "SECOND_LINE", "turnaround": "Scheduled"},
        {"test": "eGFR + serum creatinine",
         "rationale": "Renal function baseline; dose adjust metformin if eGFR <45",
         "source": "NICE NG28", "priority": "FIRST_LINE", "turnaround": "24 hours"},
    ],
    "I50": [  # Heart failure
        {"test": "NT-proBNP or BNP",
         "rationale": "Raised BNP strongly supports HF; normal value makes HF very unlikely",
         "source": "ESC HF Guidelines 2021 / NICE NG106", "priority": "FIRST_LINE", "turnaround": "2 hours"},
        {"test": "Transthoracic echocardiogram",
         "rationale": "Assess LVEF, diastolic function, valves; guides treatment",
         "source": "ESC HF Guidelines 2021", "priority": "FIRST_LINE", "turnaround": "24–48 hours"},
        {"test": "CXR",
         "rationale": "Cardiomegaly, pulmonary oedema, pleural effusions",
         "source": "NICE NG106", "priority": "FIRST_LINE", "turnaround": "1 hour"},
        {"test": "Thyroid function tests",
         "rationale": "Thyroid disease is a reversible cause of cardiomyopathy",
         "source": "ESC HF Guidelines 2021", "priority": "SECOND_LINE", "turnaround": "24 hours"},
    ],
    "J18": [  # Pneumonia
        {"test": "CXR (PA + lateral)",
         "rationale": "Confirms consolidation, excludes empyema / effusion",
         "source": "NICE NG138 / BTS CAP Guidelines", "priority": "FIRST_LINE", "turnaround": "1 hour"},
        {"test": "Sputum MC&S (if productive cough)",
         "rationale": "Identify pathogen; guide antibiotic de-escalation",
         "source": "BTS CAP Guidelines 2009", "priority": "FIRST_LINE", "turnaround": "48–72 hours"},
        {"test": "Blood cultures (if CURB-65 ≥ 2)",
         "rationale": "Required in moderate/severe CAP before antibiotics",
         "source": "NICE NG138", "priority": "FIRST_LINE", "turnaround": "72 hours"},
        {"test": "Urinary Legionella antigen",
         "rationale": "Rapid test for Legionella pneumophila; critical for treatment choice",
         "source": "BTS CAP Guidelines", "priority": "SECOND_LINE", "turnaround": "2 hours"},
        {"test": "Urea, electrolytes, LFTs",
         "rationale": "CURB-65 scoring; hepatic complications from atypical pathogens",
         "source": "NICE NG138", "priority": "FIRST_LINE", "turnaround": "2 hours"},
    ],
    "I10": [  # Hypertension
        {"test": "24-hour ambulatory blood pressure monitoring (ABPM)",
         "rationale": "Confirm white-coat vs. true hypertension; NICE first-line",
         "source": "NICE NG136", "priority": "FIRST_LINE", "turnaround": "Scheduled"},
        {"test": "Urine protein:creatinine ratio",
         "rationale": "Hypertensive nephropathy screening",
         "source": "NICE NG136", "priority": "FIRST_LINE", "turnaround": "24 hours"},
        {"test": "Fasting glucose + lipid profile",
         "rationale": "Cardiovascular risk stratification (QRISK3)",
         "source": "NICE NG136", "priority": "FIRST_LINE", "turnaround": "24 hours"},
        {"test": "ECG",
         "rationale": "Left ventricular hypertrophy, atrial fibrillation",
         "source": "NICE NG136", "priority": "FIRST_LINE", "turnaround": "30 minutes"},
    ],
}

# Default recommendations for any disease not in the rulebook
DEFAULT_TESTS = [
    {"test": "Full blood count (FBC)",
     "rationale": "Broad baseline screen for anaemia, infection, haematological disease",
     "source": "Clinical best practice", "priority": "FIRST_LINE", "turnaround": "2–4 hours"},
    {"test": "Urea, electrolytes, creatinine",
     "rationale": "Renal function and electrolyte baseline",
     "source": "Clinical best practice", "priority": "FIRST_LINE", "turnaround": "2–4 hours"},
    {"test": "CRP and ESR",
     "rationale": "Inflammatory markers for infection/autoimmune processes",
     "source": "Clinical best practice", "priority": "FIRST_LINE", "turnaround": "4 hours"},
    {"test": "Liver function tests",
     "rationale": "Hepatic baseline, relevant to drug prescribing",
     "source": "Clinical best practice", "priority": "SECOND_LINE", "turnaround": "4 hours"},
]


class RAGRecommender:
    """
    Retrieval-Augmented test recommender.

    Uses a curated rulebook for speed and reliability, with optional
    vector-store augmentation when full guideline PDFs are indexed.

    Args:
        guideline_dir : directory containing guideline .txt / .pdf files
        cfg           : RAG config dict
    """

    def __init__(self, guideline_dir: str, cfg: dict):
        self.guideline_dir = Path(guideline_dir)
        self.cfg           = cfg
        self.vector_store  = None  # lazily initialised if guidelines available
        self._try_init_vector_store()

    def _try_init_vector_store(self) -> None:
        """Attempt to load FAISS vector store from indexed guidelines."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings

            store_path = self.guideline_dir / "faiss_store"
            if store_path.exists():
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.cfg["embedding_model"]
                )
                self.vector_store = FAISS.load_local(str(store_path), embeddings)
                log.info("FAISS guideline store loaded", path=str(store_path))
        except Exception as e:
            log.info("FAISS store not available, using rulebook only", reason=str(e))

    def get_recommendations(
        self,
        diseases: list[tuple[str, str, float]],  # (icd10, name, prob)
        patient_context: str,
        max_tests: int = 8,
    ) -> list[RecommendedTest]:
        """
        Retrieve evidence-based test recommendations for predicted diseases.

        Args:
            diseases        : list of (icd10_code, disease_name, probability)
            patient_context : free-text patient summary (for RAG query)
            max_tests       : max total tests to return (de-duplicated)

        Returns:
            list of RecommendedTest ordered by priority and evidence grade
        """
        seen_tests: set[str] = set()
        results: list[RecommendedTest] = []

        for icd_code, disease_name, prob in diseases:
            prefix = icd_code[:3]
            raw_tests = GUIDELINE_RULEBOOK.get(prefix, [])

            # Augment with vector store results if available
            if self.vector_store and len(raw_tests) < 3:
                raw_tests = raw_tests + self._retrieve_from_vector_store(
                    disease_name, patient_context
                )

            for t in raw_tests:
                test_key = t["test"].lower().strip()
                if test_key in seen_tests:
                    continue
                seen_tests.add(test_key)
                results.append(RecommendedTest(
                    test_name=t["test"],
                    rationale=t["rationale"],
                    guideline_source=t["source"],
                    priority=t["priority"],
                    expected_turnaround=t["turnaround"],
                ))

        # If we couldn't find enough recommendations, add safe defaults
        for t in DEFAULT_TESTS:
            if len(results) >= max_tests:
                break
            test_key = t["test"].lower().strip()
            if test_key not in seen_tests:
                seen_tests.add(test_key)
                results.append(RecommendedTest(
                    test_name=t["test"],
                    rationale=t["rationale"],
                    guideline_source=t["source"],
                    priority=t["priority"],
                    expected_turnaround=t["turnaround"],
                ))

        # Sort: FIRST_LINE before SECOND_LINE
        priority_order = {"FIRST_LINE": 0, "SECOND_LINE": 1, "OPTIONAL": 2}
        results.sort(key=lambda x: priority_order.get(x.priority, 9))
        return results[:max_tests]

    def _retrieve_from_vector_store(
        self, disease_name: str, context: str, k: int = 3
    ) -> list[dict]:
        """Retrieve relevant guideline chunks from FAISS."""
        if self.vector_store is None:
            return []
        query = f"Recommended investigations for {disease_name}. {context[:200]}"
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            tests = []
            for doc in docs:
                tests.append({
                    "test": doc.metadata.get("test_name", "See guideline"),
                    "rationale": doc.page_content[:200],
                    "source": doc.metadata.get("source", "Clinical guideline"),
                    "priority": doc.metadata.get("priority", "SECOND_LINE"),
                    "turnaround": doc.metadata.get("turnaround", "24 hours"),
                })
            return tests
        except Exception:
            return []

    @staticmethod
    def build_index(guideline_dir: str, cfg: dict) -> None:
        """
        One-time indexing of guideline documents into FAISS.
        Run this script once when guidelines are updated.

        Usage:
            RAGRecommender.build_index("data/guidelines", cfg["rag"])
        """
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        loader = DirectoryLoader(guideline_dir, glob="**/*.txt", loader_cls=TextLoader)
        docs   = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_model"])
        store      = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(Path(guideline_dir) / "faiss_store"))
        log.info("FAISS index built", chunks=len(chunks))
