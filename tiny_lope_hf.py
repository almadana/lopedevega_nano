#!/usr/bin/env python3
"""Construye un DatasetDict (Hugging Face) desde lope3all.txt.

Genera un único ejemplo por split (train/validation/test) con el texto completo,
siguiendo el mismo particionado 90/5/5 que el builder tipo tiny_shakespeare.

Salida: una carpeta (por defecto `lope_hf_dataset/`) compatible con:
  - datasets.load_from_disk(...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _read_text(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8")
  except FileNotFoundError as e:
    raise FileNotFoundError(f"No existe el archivo: {path}") from e


def _split_90_5_5(text: str) -> dict[str, str]:
  i = int(len(text) * 0.9)
  train_text, rest = text[:i], text[i:]
  j = int(len(rest) * 0.5)
  val_text, test_text = rest[:j], rest[j:]
  return {"train": train_text, "validation": val_text, "test": test_text}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Construye y guarda un DatasetDict desde lope3all.txt.",
  )
  parser.add_argument(
      "--input",
      default="lope3all.txt",
      help="Ruta al corpus de entrada (UTF-8). Por defecto: lope3all.txt",
  )
  parser.add_argument(
      "--out",
      default="lope_hf_dataset",
      help="Directorio de salida para save_to_disk(). Por defecto: lope_hf_dataset",
  )
  args = parser.parse_args(argv)

  try:
    from datasets import Dataset, DatasetDict  # type: ignore
  except Exception as e:
    print(
        "Falta la dependencia 'datasets'. Instálala con:\n"
        "  pip install datasets\n",
        file=sys.stderr,
    )
    raise e

  root = Path(__file__).resolve().parent
  in_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
  out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

  text = _read_text(in_path)
  splits = _split_90_5_5(text)

  ds = DatasetDict(
      {
          "train": Dataset.from_dict({"text": [splits["train"]]}),
          "validation": Dataset.from_dict({"text": [splits["validation"]]}),
          "test": Dataset.from_dict({"text": [splits["test"]]}),
      }
  )

  out_dir.mkdir(parents=True, exist_ok=True)
  ds.save_to_disk(str(out_dir))

  print(f"OK. Dataset guardado en: {out_dir}")
  print("Splits:", ", ".join(ds.keys()))
  print("Tamaños (chars):", {k: len(v) for k, v in splits.items()})
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

