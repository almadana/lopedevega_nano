# coding=utf-8
# Copyright 2026 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Corpus de texto en formato similar a Tiny Shakespeare (un ejemplo por split).

Los datos se leen de ``lope3all.txt`` en el mismo directorio que este módulo,
en lugar del ``input.txt`` de tinyshakespeare en el repositorio de Karpathy.
"""

from __future__ import annotations

import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """Builder de dataset de texto a partir de ``lope3all.txt`` local."""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description='Texto completo de lope3all.txt partido en train/validation/test.',
        features=tfds.features.FeaturesDict({'text': tfds.features.Text()}),
        supervised_keys=None,
        homepage='file://lope3all.txt (local)',
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    del dl_manager  # No hay descarga remota; el corpus es local.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, 'lope3all.txt')
    if not tf.io.gfile.exists(txt_path):
      raise FileNotFoundError(
          f'No se encuentra el corpus en {txt_path}. '
          'Coloca lope3all.txt junto a tiny_shakespeare_dataset_builder.py.'
      )
    with open(txt_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # 90/5/5 split
    i = int(len(text) * 0.9)
    train_text, text = text[:i], text[i:]
    i = int(len(text) * 0.5)
    validation_text, text = text[:i], text[i:]
    test_text = text

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'split_key': 'train', 'split_text': train_text},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'split_key': 'validation',
                'split_text': validation_text,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={'split_key': 'test', 'split_text': test_text},
        ),
    ]

  def _generate_examples(self, split_key, split_text):
    """Yields examples."""
    data_key = split_key  # Should uniquely identify the thing yielded
    feature_dict = {'text': split_text}
    yield data_key, feature_dict
