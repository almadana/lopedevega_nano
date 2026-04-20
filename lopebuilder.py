#!/usr/bin/env python3
"""Prepara el dataset TFDS leyendo lope3all.txt (mismo directorio que este script)."""

from __future__ import annotations

import os
import sys


def main() -> None:
  root = os.path.dirname(os.path.abspath(__file__))
  if root not in sys.path:
    sys.path.insert(0, root)

  # Menos ruido de TensorFlow en consola (warnings de oneDNN, etc.)
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

  from tiny_shakespeare_dataset_builder import Builder

  data_dir = os.path.join(root, 'tfds_data')
  print(f'Generando dataset en: {data_dir}')
  builder = Builder(data_dir=data_dir)
  builder.download_and_prepare()
  splits = list(builder.info.splits.keys())
  print('Hecho. Splits:', ', '.join(splits))


if __name__ == '__main__':
  main()
