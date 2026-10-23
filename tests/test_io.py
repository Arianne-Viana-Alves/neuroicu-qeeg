"""Testes da camada de leitura com dados sintéticos no formato do I-CARE."""

from pathlib import Path

import numpy as np
import pytest
import wfdb

from neuroicu.io import (
    build_index,
    discover,
    load_clinical,
    load_recording,
    resolve_data_dir,
    to_bipolar,
)

FS = 128
CHANNELS = ["Fp1", "F7", "T3", "Cz", "Pz"]
CLINICAL = """Patient: 0284
Hospital: A
Age: 53
Sex: Male
ROSC: 12
OHCA: True
Shockable Rhythm: False
TTM: 33
Outcome: Poor
CPC: 5
"""


def _write_segment(folder: Path, name: str, seconds: int = 10, group_channels=CHANNELS):
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 20, size=(seconds * FS, len(group_channels)))
    wfdb.wrsamp(
        name,
        fs=FS,
        units=["uV"] * len(group_channels),
        sig_name=list(group_channels),
        p_signal=signal,
        fmt=["16"] * len(group_channels),
        write_dir=str(folder),
    )


@pytest.fixture
def icare_dir(tmp_path: Path) -> Path:
    patient = tmp_path / "0284"
    patient.mkdir()
    (patient / "0284.txt").write_text(CLINICAL, encoding="utf-8")
    _write_segment(patient, "0284_001_004_EEG")
    _write_segment(patient, "0284_002_005_EEG")
    _write_segment(patient, "0284_003_080_EEG")  # depois de 72 h
    _write_segment(patient, "0284_001_004_ECG", group_channels=["ECG"])
    return tmp_path


def test_discover_keeps_only_eeg(icare_dir):
    recs = discover(icare_dir)
    assert [r.hour for r in recs] == [4, 5, 80]
    assert all(r.group == "EEG" and r.patient_id == "0284" for r in recs)


def test_discover_max_hour(icare_dir):
    assert [r.hour for r in discover(icare_dir, max_hour=72)] == [4, 5]


def test_load_recording_shape_and_units(icare_dir):
    rec = load_recording(discover(icare_dir)[0])
    assert rec.data.shape == (len(CHANNELS), 10 * FS)
    assert rec.fs == FS
    assert rec.ch_names == CHANNELS
    assert 5 < rec.data.std() < 50  # continua em microvolts
    assert rec.duration_s == pytest.approx(10)


def test_load_recording_channel_selection(icare_dir):
    rec = load_recording(discover(icare_dir)[0], channels=["Cz", "Fp1"])
    assert rec.ch_names == ["Cz", "Fp1"]
    with pytest.raises(KeyError):
        load_recording(discover(icare_dir)[0], channels=["O2"])


def test_to_bipolar_skips_missing_pairs(icare_dir):
    rec = load_recording(discover(icare_dir)[0])
    bip = to_bipolar(rec, [("Fp1", "F7"), ("Cz", "Pz"), ("O1", "O2")])
    assert bip.ch_names == ["Fp1-F7", "Cz-Pz"]
    np.testing.assert_allclose(bip.data[0], rec.data[0] - rec.data[1])


def test_load_clinical_types(icare_dir):
    clinical = load_clinical(icare_dir / "0284")
    assert clinical["Patient"] == "0284"  # zero à esquerda preservado
    assert clinical["Age"] == 53
    assert clinical["OHCA"] is True
    assert clinical["Outcome"] == "Poor"
    assert clinical["CPC"] == 5


def test_build_index_joins_clinical(icare_dir):
    df = build_index(icare_dir, max_hour=72)
    assert len(df) == 2
    assert set(df["Outcome"]) == {"Poor"}


def test_generic_folder_without_icare_names(tmp_path):
    folder = tmp_path / "paciente_A"
    folder.mkdir()
    _write_segment(folder, "registro_qualquer")
    (rec,) = discover(tmp_path)
    assert rec.patient_id == "paciente_A" and rec.hour is None


def test_edf_support(tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    info = mne.create_info(CHANNELS, FS, ch_types="eeg")
    data = np.random.default_rng(1).normal(0, 20e-6, size=(len(CHANNELS), 10 * FS))
    folder = tmp_path / "paciente_B"
    folder.mkdir()
    mne.io.RawArray(data, info, verbose="error").export(
        folder / "exame.edf", verbose="error"
    )
    rec = load_recording(discover(tmp_path)[0])
    assert rec.info.fmt == "edf"
    assert 5 < rec.data.std() < 50  # convertido de volts para microvolts


def test_missing_data_dir_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="NEUROICU_DATA_DIR"):
        resolve_data_dir(tmp_path / "nao_existe")
