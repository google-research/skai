"""Tests for cloud_postprocess_lib."""

from absl.testing import absltest
import pandas as pd
from skai.model import cloud_postprocess_lib


class CloudPostprocessLibTest(absltest.TestCase):

  def test_raise_error_on_no_cloud_score(self):
    with self.assertRaisesRegex(ValueError, "no `cloud_score` column found"):
      cloud_postprocess_lib.identify_clouds(
          pd.DataFrame(), distance_threshold=0.5
      )

  def test_raise_error_on_no_longitude_or_latitude(self):
    with self.assertRaisesRegex(
        ValueError, "no `longitude` or `latitude` column found"
    ):
      cloud_postprocess_lib.identify_clouds(
          pd.DataFrame(
              {"cloud_score": [0.1, 0.2, 0.3], "int64_id": [0, 1, 2]}
          ),
          distance_threshold=0.5,
      )

  def test_raise_error_on_no_int64_id(self):
    with self.assertRaisesRegex(ValueError, "no `int64_id` column found"):
      cloud_postprocess_lib.identify_clouds(
          pd.DataFrame({
              "longitude": [10, 10, 10, 10],
              "latitude": [10, 10, 10, 10],
              "cloud_score": [0.1, 0.2, 0.3, 0.4],
          }),
          distance_threshold=0.5,
      )

  def test_convert_to_geopandas_projection(self):
    df = pd.DataFrame({
        "longitude": [10, 10, 10, 10],
        "latitude": [10, 10, 10, 10],
        "cloud_score": [0.1, 0.2, 0.3, 0.4],
    })
    gdf = cloud_postprocess_lib.convert_to_geopandas(df)
    self.assertEqual(str(gdf.crs), "EPSG:32632")

  def test_identify_clouds_based_cloud_score_only(self):
    df = pd.DataFrame({
        "int64_id": [0, 1, 2, 3],
        "longitude": [10, 20, 30, 50],
        "latitude": [10, 20, 40, 50],
        "cloud_score": [0.1, 0.6, 0.1, 0.7],
    })
    df = cloud_postprocess_lib.identify_clouds(df, distance_threshold=500)
    self.assertSequenceEqual(df["is_cloudy"].tolist(), [0, 1, 0, 1])

  def test_identify_clouds_based_on_cloud_score_and_distance(self):
    df = pd.DataFrame({
        "int64_id": [0, 1, 2, 3],
        "longitude": [10, 10.001, 30, 50],
        "latitude": [10, 10.001, 40, 50],
        "cloud_score": [0.1, 0.6, 0.1, 0.7],
    })
    df = cloud_postprocess_lib.identify_clouds(df, distance_threshold=500)
    self.assertSequenceEqual(df["is_cloudy"].tolist(), [1, 1, 0, 1])


if __name__ == "__main__":
  absltest.main()
