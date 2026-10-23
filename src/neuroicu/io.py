"""Leitura genérica de EEG a partir de uma pasta local.

O código não baixa nada: você aponta uma pasta (``data_dir``) e ele descobre
sozinho os registros. Formatos suportados:

- WFDB (``.hea`` + ``.mat``/``.dat``), usado pelo I-CARE / PhysioNet
- EDF/BDF, usado pela maioria dos equipamentos clínicos

Para o I-CARE, a estrutura esperada é a do PhysioNet::

    data_dir/
    └── 0284/
        ├── 0284.txt                 # dados clínicos e desfecho
        ├── 0284_001_004_EEG.hea     # segmento 001, hora 004 após a parada
        └── 0284_001_004_EEG.mat

Qualquer outra organização também funciona: o paciente é a pasta-mãe do
arquivo, e o que não seguir o padrão de nome do I-CARE fica sem hora/segmento.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENV_DATA_DIR = "NEUROICU_DATA_DIR"

# 0284_001_004_EEG  ->  paciente 0284, segmento 001, hora 004, grupo EEG
ICARE_NAME = re.compile(
    r"^(?P<patient>\d+)_(?P<segment>\d+)_(?P<hour>\d+)_(?P<group>[A-Z]+)$"
)

EDF_SUFFIXES = {".edf", ".bdf"}


@dataclass(frozen=True)
class RecordingInfo:
    """Um arquivo de EEG encontrado no disco (ainda não carregado)."""

    path: Path
    patient_id: str
    fmt: str  # "wfdb" ou "edf"
    segment: int | None = None
    hour: int | None = None
    group: str | None = None


@dataclass
class EEGRecording:
    """Sinal carregado, sempre em microvolts e no formato (canais, amostras)."""

    data: np.ndarray
    fs: float
    ch_names: list[str]
    info: RecordingInfo
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Pasta de dados: argumento > variável NEUROICU_DATA_DIR > ./data/raw."""
    if data_dir is None:
        data_dir = os.environ.get(ENV_DATA_DIR, "data/raw")
    path = Path(data_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Pasta de dados não encontrada: {path}. "
            f"Passe o caminho ou defina {ENV_DATA_DIR}."
        )
    return path


def _info_from_path(path: Path, fmt: str) -> RecordingInfo:
    match = ICARE_NAME.match(path.stem)
    if match:
        return RecordingInfo(
            path=path,
            patient_id=match["patient"],
            fmt=fmt,
            segment=int(match["segment"]),
            hour=int(match["hour"]),
            group=match["group"],
        )
    return RecordingInfo(path=path, patient_id=path.parent.name, fmt=fmt)


def discover(
    data_dir: str | os.PathLike | None = None,
    groups: tuple[str, ...] | None = ("EEG",),
    max_hour: int | None = None,
) -> list[RecordingInfo]:
    """Lista os registros de EEG encontrados em ``data_dir`` (recursivo).

    Args:
        data_dir: pasta raiz. Se None, usa NEUROICU_DATA_DIR ou ./data/raw.
        groups: grupos do I-CARE a manter (EEG, ECG, REF, OTHER). None = todos.
            Arquivos fora do padrão I-CARE são sempre mantidos.
        max_hour: descarta segmentos que começam depois dessa hora
            (ex.: 72 para usar só as primeiras 72 h após a parada).
    """
    root = resolve_data_dir(data_dir)
    found: list[RecordingInfo] = []
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".hea":
            info = _info_from_path(path.with_suffix(""), "wfdb")
        elif suffix in EDF_SUFFIXES:
            info = _info_from_path(path, "edf")
        else:
            continue
        if groups is not None and info.group is not None and info.group not in groups:
            continue
        if max_hour is not None and info.hour is not None and info.hour > max_hour:
            continue
        found.append(info)
    return found


def load_recording(
    info: RecordingInfo, channels: list[str] | None = None
) -> EEGRecording:
    """Carrega um registro em microvolts. ``channels`` filtra e ordena canais."""
    if info.fmt == "wfdb":
        data, fs, names, meta = _load_wfdb(info.path)
    elif info.fmt == "edf":
        data, fs, names, meta = _load_edf(info.path)
    else:
        raise ValueError(f"Formato não suportado: {info.fmt}")

    if channels is not None:
        missing = [c for c in channels if c not in names]
        if missing:
            raise KeyError(f"Canais ausentes em {info.path.name}: {missing}")
        idx = [names.index(c) for c in channels]
        data, names = data[idx], list(channels)

    return EEGRecording(data=data, fs=fs, ch_names=names, info=info, meta=meta)


_TO_MICROVOLT = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


def _load_wfdb(record_path: Path) -> tuple[np.ndarray, float, list[str], dict]:
    import wfdb

    rec = wfdb.rdrecord(str(record_path))
    data = rec.p_signal.T.astype(np.float32)
    for i, unit in enumerate(rec.units or []):
        scale = _TO_MICROVOLT.get(str(unit).strip().lower())
        if scale is not None and scale != 1.0:
            data[i] *= scale
    meta = {"units_original": list(rec.units or []), "comments": list(rec.comments or [])}
    for comment in rec.comments or []:
        if ":" in comment:
            key, value = comment.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return data, float(rec.fs), list(rec.sig_name), meta


def _load_edf(path: Path) -> tuple[np.ndarray, float, list[str], dict]:
    import mne

    raw = mne.io.read_raw(str(path), preload=True, verbose="error")
    data = (raw.get_data() * 1e6).astype(np.float32)  # MNE usa volts
    meta = {"meas_date": str(raw.info["meas_date"])}
    return data, float(raw.info["sfreq"]), list(raw.ch_names), meta


def to_bipolar(rec: EEGRecording, pairs: list[tuple[str, str]]) -> EEGRecording:
    """Monta derivações bipolares (ex.: Fp1-F7) a partir de canais referenciais.

    Pares com algum canal ausente são ignorados, já que no I-CARE nem todo
    paciente tem todos os eletrodos.
    """
    idx = {name: i for i, name in enumerate(rec.ch_names)}
    rows, names = [], []
    for a, b in pairs:
        if a in idx and b in idx:
            rows.append(rec.data[idx[a]] - rec.data[idx[b]])
            names.append(f"{a}-{b}")
    if not rows:
        raise ValueError(f"Nenhum par bipolar disponível em {rec.info.path.name}")
    return EEGRecording(
        data=np.vstack(rows), fs=rec.fs, ch_names=names, info=rec.info, meta=rec.meta
    )


def _parse_value(text: str):
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"nan", ""}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def load_clinical(patient_dir: str | os.PathLike) -> dict:
    """Lê o ``<id>.txt`` do I-CARE (linhas ``Chave: valor``) como dicionário."""
    patient_dir = Path(patient_dir)
    txt = patient_dir / f"{patient_dir.name}.txt"
    if not txt.is_file():
        return {}
    out: dict = {}
    for line in txt.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        # IDs ficam como texto para não perder o zero à esquerda (0284)
        out[key] = value if key.lower() in {"patient", "hospital"} else _parse_value(value)
    return out


def build_index(
    data_dir: str | os.PathLike | None = None,
    groups: tuple[str, ...] | None = ("EEG",),
    max_hour: int | None = None,
) -> pd.DataFrame:
    """Tabela com um registro por linha + dados clínicos do paciente.

    Não carrega os sinais, então é rápida mesmo com muitos pacientes.
    """
    recordings = discover(data_dir, groups=groups, max_hour=max_hour)
    clinical_cache: dict[Path, dict] = {}
    rows = []
    for info in recordings:
        folder = info.path.parent
        if folder not in clinical_cache:
            clinical_cache[folder] = load_clinical(folder)
        rows.append(
            {
                "patient_id": info.patient_id,
                "segment": info.segment,
                "hour": info.hour,
                "group": info.group,
                "format": info.fmt,
                "path": str(info.path),
                **clinical_cache[folder],
            }
        )
    columns = ["patient_id", "segment", "hour", "group", "format", "path"]
    return pd.DataFrame(rows, columns=None if rows else columns)
