"""Gera o índice de registros de uma pasta local.

Uso:
    python -m neuroicu.index --data-dir /caminho/para/i-care/training
    python -m neuroicu.index --max-hour 72 --out data/processed/index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroicu.io import build_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", default=None, help="pasta dos dados (padrão: NEUROICU_DATA_DIR ou data/raw)"
    )
    parser.add_argument(
        "--max-hour", type=int, default=None, help="usar só segmentos até esta hora após a parada"
    )
    parser.add_argument("--out", default="data/processed/index.csv", help="CSV de saída")
    args = parser.parse_args(argv)

    df = build_index(args.data_dir, max_hour=args.max_hour)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_patients = df["patient_id"].nunique() if len(df) else 0
    print(f"{len(df)} registros de {n_patients} pacientes -> {out}")
    if "Outcome" in df.columns and n_patients:
        outcomes = df.drop_duplicates("patient_id")["Outcome"].value_counts(dropna=False)
        print("Desfecho por paciente:")
        print(outcomes.to_string())


if __name__ == "__main__":
    main()
