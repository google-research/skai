"""Tests for the xm_vlm_zero_shot_vertex_lib module.

Specifically focuses on the __ensemble_prediction_csvs function.
"""

import os

from absl.testing import absltest
import pandas as pd
from skai.model import xm_vlm_zero_shot_vertex_lib


class CombineEnsembleOutputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def test_damage_score_is_mean(self):
    """Tests that the damage_score is the mean of the ensemble.

    Also tests that the resulting 'label' and 'damage' columns are correct.
    """
    csv_data0 = {
        'example_id': ['A', 'B', 'C'],
        'damage_score': [0.2, 0.4, 0.8],
        'is_cloudy': [0.2, 0.4, 0.8],
        'longitude': [10, 10, 20],
        'latitude': [20, 20, 30],
        'building_id': [1, 1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 456, 789],
        'label': [0, 0, 1],
        'damage': [False, False, True],
    }
    csv_data1 = {
        'example_id': ['A', 'B', 'C'],
        'damage_score': [0.6, 1.0, 1.0],
        'longitude': [10, 10, 20],
        'latitude': [20, 20, 30],
        'building_id': [1, 1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 456, 789],
        'label': [0, 1, 1],
        'damage': [False, True, True],
    }
    pd.DataFrame(csv_data0).to_csv(
        os.path.join(self.temp_dir, 'model0_output.csv'), index=False
    )
    pd.DataFrame(csv_data1).to_csv(
        os.path.join(self.temp_dir, 'model1_output.csv'), index=False
    )
    combined_df = xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([
        os.path.join(self.temp_dir, 'model0_output.csv'),
        os.path.join(self.temp_dir, 'model1_output.csv'),
    ]).set_index('example_id', drop=False).sort_index()
    expected_data = {
        'example_id': ['A', 'B', 'C'],
        'damage_score': [0.4, 0.7, 0.9],
        'is_cloudy': [0.2, 0.4, 0.8],
        'longitude': [10, 10, 20],
        'latitude': [20, 20, 30],
        'building_id': [1, 1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 456, 789],
        'label': [-1., -1., -1.],
        'damage': [False, True, True],
        'damage_score_0': [0.2, 0.4, 0.8],
        'damage_score_1': [0.6, 1.0, 1.0],
    }
    expected_df = pd.DataFrame(expected_data).set_index(
        'example_id', drop=False
    ).reindex(combined_df.columns, axis=1)
    pd.testing.assert_frame_equal(combined_df, expected_df)

  def test_missing_example_ids(self):
    csv_data0 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.2, 0.8],
        'is_cloudy': [0.2, 0.4],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [0, 1],
        'damage': [False, True],
    }
    csv_data1 = {
        'example_id': ['A', 'C'],
        'damage_score': [0.6, 0.5],
        'longitude': [10, 12],
        'latitude': [20, 22],
        'building_id': [1, 3],
        'plus_code': ['AAAA+AA', 'CCCC+CC'],
        'int64_id': [123, 789],
        'label': [0, 0],
        'damage': [False, False],
    }
    file_paths = [
        os.path.join(self.temp_dir, 'model0_output.csv'),
        os.path.join(self.temp_dir, 'model1_output.csv'),
    ]
    pd.DataFrame(csv_data0).to_csv(file_paths[0], index=False)
    pd.DataFrame(csv_data1).to_csv(file_paths[1], index=False)
    with self.assertRaisesRegex(
        ValueError,
        f'The CSV of {file_paths[1]} does not contain the same set of '
        f'example_ids as the SigLIP CSV at {file_paths[0]}.',
    ):
      _ = (
          xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs(file_paths)
          .set_index('example_id', drop=False)
          .sort_index()
      )

  def test_missing_columns(self):
    # Only 1 CSV has the 'is_cloudy' column.
    csv_data0 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.2, 0.8],
        'is_cloudy': [0.1, 0.7],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [0, 1],
        'damage': [False, True],
    }
    csv_data1 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.6, 0.2],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [0, 1],
        'damage': [False, True],
    }
    pd.DataFrame(csv_data0).to_csv(
        os.path.join(self.temp_dir, 'model0_output.csv'), index=False
    )
    pd.DataFrame(csv_data1).to_csv(
        os.path.join(self.temp_dir, 'model1_output.csv'), index=False
    )
    combined_df = xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([
        os.path.join(self.temp_dir, 'model0_output.csv'),
        os.path.join(self.temp_dir, 'model1_output.csv'),
    ]).set_index('example_id', drop=False).sort_index()
    expected_data = {
        'example_id': ['A', 'B'],
        'damage_score': [0.4, 0.5],
        'is_cloudy': [0.1, 0.7],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [-1., -1.],
        'damage': [False, False],
        'damage_score_0': [0.2, 0.8],
        'damage_score_1': [0.6, 0.2],
    }
    expected_df = pd.DataFrame(expected_data).set_index(
        'example_id', drop=False
    ).reindex(combined_df.columns, axis=1)
    pd.testing.assert_frame_equal(combined_df, expected_df)

  def test_more_than_two_csvs(self):
    csv_data0 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.2, 0.8],
        'is_cloudy': [0.1, 0.7],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [0, 1],
        'damage': [False, True],
    }
    csv_data1 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.6, 0.2],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [0, 1],
        'damage': [False, True],
    }
    csv_data2 = {
        'example_id': ['A', 'B'],
        'damage_score': [0.1, 0.2],
        'longitude': [20, 20],
        'latitude': [30, 10],
        'building_id': [2, 3],
        'plus_code': ['BBBB+BB', 'CCCC+CC'],
        'int64_id': [456, 789],
        'label': [1, 1],
        'damage': [True, False],
    }
    pd.DataFrame(csv_data0).to_csv(
        os.path.join(self.temp_dir, 'model0_output.csv'), index=False
    )
    pd.DataFrame(csv_data1).to_csv(
        os.path.join(self.temp_dir, 'model1_output.csv'), index=False
    )
    pd.DataFrame(csv_data2).to_csv(
        os.path.join(self.temp_dir, 'model2_output.csv'), index=False
    )
    combined_df = xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([
        os.path.join(self.temp_dir, 'model0_output.csv'),
        os.path.join(self.temp_dir, 'model1_output.csv'),
        os.path.join(self.temp_dir, 'model2_output.csv'),
    ]).set_index('example_id', drop=False).sort_index()
    expected_data = {
        'example_id': ['A', 'B'],
        'damage_score': [0.3, 0.4],
        'is_cloudy': [0.1, 0.7],
        'longitude': [10, 20],
        'latitude': [20, 30],
        'building_id': [1, 2],
        'plus_code': ['AAAA+AA', 'BBBB+BB'],
        'int64_id': [123, 456],
        'label': [-1., -1.],
        'damage': [False, False],
        'damage_score_0': [0.2, 0.8],
        'damage_score_1': [0.6, 0.2],
        'damage_score_2': [0.1, 0.2],
    }
    expected_df = pd.DataFrame(expected_data).set_index(
        'example_id', drop=False
    ).reindex(combined_df.columns, axis=1)
    pd.testing.assert_frame_equal(combined_df, expected_df)

  def test_duplicate_example_ids(self):
    # Create sample CSV files with duplicate example_ids
    csv_data0 = {
        'example_id': ['A', 'A', 'B', 'C'],
        'damage_score': [0.1, 0.2, 0.8, 0.9],
        'is_cloudy': [0.2, 0.4, 0.8, 0.2],
        'longitude': [10, 11, 20, 15],
        'latitude': [20, 21, 30, 25],
        'building_id': [1, 1, 2, 4],
        'plus_code': ['AAAA+AA', 'AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 123, 456, 999],
        'label': [0, 0, 1, 1],
        'damage': [False, False, True, True],
    }
    csv_data1 = {
        'example_id': ['A', 'B', 'B', 'C'],
        'damage_score': [0.6, 0.4, 1.0, 1.0],
        'longitude': [10, 20, 22, 15],
        'latitude': [20, 30, 31, 25],
        'building_id': [1, 2, 2, 4],
        'plus_code': ['AAAA+AA', 'BBBB+BB', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 456, 457, 999],
        'label': [0, 1, 1, 1],
        'damage': [False, True, True, True],
    }
    pd.DataFrame(csv_data0).to_csv(
        os.path.join(self.temp_dir, 'model0_output.csv'), index=False
    )
    pd.DataFrame(csv_data1).to_csv(
        os.path.join(self.temp_dir, 'model1_output.csv'), index=False
    )
    combined_df = xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([
        os.path.join(self.temp_dir, 'model0_output.csv'),
        os.path.join(self.temp_dir, 'model1_output.csv'),
    ]).set_index('example_id', drop=False).sort_index()
    expected_data = {
        'example_id': ['A', 'B', 'C'],
        'damage_score': [0.35, 0.6, 0.95],
        'is_cloudy': [0.2, 0.8, 0.2],
        'longitude': [10, 20, 15],
        'latitude': [20, 30, 25],
        'building_id': [1, 2, 4],
        'plus_code': ['AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
        'int64_id': [123, 456, 999],
        'label': [-1.0, -1.0, -1.0],
        'damage': [False, True, True],
        'damage_score_0': [0.1, 0.8, 0.9],
        'damage_score_1': [0.6, 0.4, 1.0],
    }
    expected_df = (
        pd.DataFrame(expected_data)
        .set_index('example_id', drop=False)
        .reindex(combined_df.columns, axis=1)
    )
    pd.testing.assert_frame_equal(combined_df, expected_df)

  def test_no_files(self):
    with self.assertRaisesRegex(
        ValueError,
        'At least one CSV file must be provided, specifically '
        'the SigLIP model output.',
    ):
      xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([])

  def test_siglip_csv_not_first(self):
    with self.assertRaisesRegex(
        ValueError,
        'First CSV in file_paths must be the SigLIP model output, which must '
        'contain is_cloudy, example_id, and damage_score columns.',
    ):
      csv_data0 = {
          'example_id': ['A', 'B', 'B', 'C'],
          'damage_score': [0.6, 0.4, 1.0, 1.0],
          'longitude': [10, 20, 22, 15],
          'latitude': [20, 30, 31, 25],
          'building_id': [1, 2, 2, 4],
          'plus_code': ['AAAA+AA', 'BBBB+BB', 'BBBB+BB', 'CCCC+CC'],
          'int64_id': [123, 456, 457, 999],
          'label': [0, 1, 1, 1],
          'damage': [False, True, True, True],
      }
      csv_data1 = {
          'example_id': ['A', 'A', 'B', 'C'],
          'damage_score': [0.1, 0.2, 0.8, 0.9],
          'is_cloudy': [0.2, 0.4, 0.8, 0.2],
          'longitude': [10, 11, 20, 15],
          'latitude': [20, 21, 30, 25],
          'building_id': [1, 1, 2, 4],
          'plus_code': ['AAAA+AA', 'AAAA+AA', 'BBBB+BB', 'CCCC+CC'],
          'int64_id': [123, 123, 456, 999],
          'label': [0, 0, 1, 1],
          'damage': [False, False, True, True],
      }
      pd.DataFrame(csv_data0).to_csv(
          os.path.join(self.temp_dir, 'model0_output.csv'), index=False
      )
      pd.DataFrame(csv_data1).to_csv(
          os.path.join(self.temp_dir, 'model1_output.csv'), index=False
      )
      _ = xm_vlm_zero_shot_vertex_lib._ensemble_prediction_csvs([
          os.path.join(self.temp_dir, 'model0_output.csv'),
          os.path.join(self.temp_dir, 'model1_output.csv'),
      ])


if __name__ == '__main__':
  absltest.main()
