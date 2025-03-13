"""Tests functions in representative_sampling.py.
"""

from absl.testing import absltest
import geopandas as gpd
import pandas as pd
from skai import representative_sampling


class SampleExamplesForLabelingLibTest(absltest.TestCase):

  def test_divide_aoi_into_grid_small_area(self):
    """Tests when the AOI is too small to be divided into 16 cells."""
    aoi_gdf = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.85050],
            [92.850449, 20.85051],
            [92.85050, 20.85051],
            [92.85050, 20.85050],
        ] * 4,
        columns=['longitude', 'latitude'],
    )
    aoi_gdf = representative_sampling._divide_aoi_into_grid(aoi_gdf)
    # The points are too close to each other to be divided into 16 separate
    # cells, so they should be categorized in the same grid cell.
    self.assertEqual(aoi_gdf['grid_cell'].value_counts()[0], 16)

  def test_divide_aoi_into_grid_empty_cells(self):
    """Tests four clusters of points with empty cells in between."""
    aoi_gdf = gpd.GeoDataFrame(
        data=[
            # Cluster 0
            [92.850449, 20.85050],
            [92.850449, 20.85051],
            [92.85050, 20.85051],
            [92.85051, 20.85050],
            # Cluster 1
            [93.85050, 20.85050],
            [93.850449, 20.85050],
            [93.850449, 20.85051],
            [93.85050, 20.85051],
            # Cluster 2
            [92.850449, 21.85050],
            [92.850449, 21.85051],
            [92.85050, 21.85051],
            [92.85050, 21.85050],
            # Cluster 3
            [93.850449, 21.85050],
            [93.850449, 21.85051],
            [93.85050, 21.85051],
            [93.85050, 21.85050],
        ],
        columns=['longitude', 'latitude'],
    )
    aoi_gdf = representative_sampling._divide_aoi_into_grid(aoi_gdf)
    self.assertLen(aoi_gdf['grid_cell'].value_counts(), 4)
    self.assertEqual(set(aoi_gdf['grid_cell'].unique()), set([0, 1, 4, 5]))

  def test_get_quartile_gdf(self):
    """Tests getting examples that belong to a given quartile from a cell."""
    cell_gdf = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.85050, 0.0],
            [92.850449, 20.85051, 0.1],
            [92.85050, 20.85051, 0.2],
            [92.85051, 20.85050, 0.3],
            [92.85051, 20.85050, 0.4],
        ],
        columns=['longitude', 'latitude', 'damage_score'],
    )
    quartiles = list(cell_gdf['damage_score'].quantile([0.25, 0.5, 0.75]))
    for i in range(4):  # 4 quartiles total.
      quartile_gdf, bucket_score_increment = (
          representative_sampling._get_quartile_gdf(
              cell_gdf, quartiles, quartile_idx=i
          )
      )
      if i == 0:
        # First quartile contains examples with score 0.0 and 0.1.
        self.assertLen(quartile_gdf, 2)
      else:
        # Other quartiles should only contain the one example in that quartile.
        self.assertLen(quartile_gdf, 1)
      self.assertEqual(round(bucket_score_increment, 3), 0.025)

  def test_get_bucket_gdf(self):
    """Tests getting examples that fall in a given bucket from a quartile."""
    quartile_gdf = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.85050, 0.01],
            [92.850449, 20.85051, 0.02],
            [92.85050, 20.85051, 0.03],
            [92.85051, 20.85050, 0.04],
            [92.850449, 20.85050, 0.05],
            [92.850449, 20.85051, 0.06],
            [92.85050, 20.85051, 0.07],
            [92.85051, 20.85050, 0.08],
        ],
        columns=['longitude', 'latitude', 'damage_score'],
    )
    bucket_gdf = representative_sampling._get_bucket_gdf(
        quartile_gdf,
        0.0,
        bucket_score_increment=0.02,
        bucket_idx=0,
        all_samples=[],
    )
    self.assertLen(bucket_gdf, 1)

  def test_drop_points_within_sample(self):
    """Tests dropping points within a buffer distance of the sampled point."""
    sample_geometry = gpd.points_from_xy(
        [92.850449], [20.85050], crs='EPSG:4326'
    ).to_crs('EPSG:32737')
    bucket_gdf = gpd.GeoDataFrame(
        gpd.GeoDataFrame(
            data=[
                [92.850449, 20.85050, 0.1],
                [92.850449, 20.85051, 0.2],
                [93.85050, 21.85051, 0.3],
                [93.85051, 21.85050, 0.4],
            ],
            columns=['longitude', 'latitude', 'damage_score'],
        ),
        geometry=gpd.points_from_xy(
            [92.850449, 92.85050, 93.85051, 93.85050],
            [20.85050, 20.85051, 21.85051, 21.85050],
        ),
        crs='EPSG:4326',
    ).to_crs('EPSG:32737')
    bucket_gdf = representative_sampling._drop_points_within_sample(
        sample_geometry, bucket_gdf, 80
    )
    self.assertLen(bucket_gdf, 2)

  def test_sample_examples(self):
    """Tests that the requested number of examples are sampled."""
    df = pd.DataFrame(
        data=[
            [92.850449, 20.148951, 0.1],
            [92.889694, 20.157515, 0.2],
            [92.889740, 20.157454, 0.3],
            [92.850479, 20.148664, 0.4],
            [92.898537, 20.150021, 0.5],
            [92.898537, 20.160021, 0.6],
            [92.868537, 20.170021, 0.7],
            [92.878537, 20.180021, 0.8],
            [92.888537, 20.190021, 0.9],
            [92.858537, 20.200021, 1.0],
        ],
        columns=['longitude', 'latitude', 'damage_score'],
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326',
    )
    gdf = representative_sampling._divide_aoi_into_grid(gdf)
    sampled_idx = representative_sampling.sample_examples(
        gdf, 5, 2, gdf['grid_cell'].unique(), 80
    )
    self.assertLen(sampled_idx, 5)

  def test_sample_examples_fewer_than_requested(self):
    """Tests when there are fewer examples available than requested."""
    gdf = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.148951, 0.1],
            [92.889694, 20.157515, 0.2],
            [92.889740, 20.157454, 0.3]
        ],
        columns=['longitude', 'latitude', 'damage_score'],
    )
    gdf = representative_sampling._divide_aoi_into_grid(gdf)
    sampled_idx = representative_sampling.sample_examples(
        gdf, 5, 1, gdf['grid_cell'].unique(), 80
    )
    self.assertEmpty(sampled_idx)

  def test_run_representative_sampling_bad_input(self):
    """Tests when there are fewer examples available than requested."""
    df = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.148951],
            [92.889694, 20.157515],
            [92.889740, 20.157454],
            [92.850479, 20.148664],
            [92.898537, 20.150021],
            [92.898537, 20.160021],
            [92.868537, 20.170021],
            [92.878537, 20.180021],
        ],
        columns=['longitude', 'latitude'],
    )
    df = representative_sampling._divide_aoi_into_grid(df)
    self.assertRaises(
        ValueError,
        representative_sampling.run_representative_sampling,
        df,
        5,
        2,
        df['grid_cell'].unique(),
        80
    )

  def test_run_representative_sampling(self):
    """Tests the run_representative_sampling function."""
    gdf = gpd.GeoDataFrame(
        data=[
            [92.850449, 20.148951, 0.11, False],
            [92.889694, 20.157515, 0.29, True],
            [92.889740, 20.157454, 0.36, True],
            [92.850479, 20.148664, 0.05, False],
            [92.898537, 20.150021, 0.99, False],
            [92.898537, 20.160021, 0.72, True],
            [92.868537, 20.170021, 0.61, True],
            [92.878537, 20.180021, 0.57, False],
            [92.898537, 20.160021, 0.48, False],
            [92.878537, 20.180021, 0.12, False],
            [92.898537, 20.160021, 0.22, False],
            [92.878537, 20.180021, 0.83, False],
            [92.898537, 20.160021, 0.89, False],
            [92.878537, 20.180021, 0.52, False],
            [92.898537, 20.160021, 0.93, False],
        ],
        columns=['longitude', 'latitude', 'damage_score', 'is_cloudy'],
    )
    train_df, test_df = representative_sampling.run_representative_sampling(
        gdf, 8, 1, 0.5, 80
    )
    self.assertLen(train_df, 4)
    self.assertLen(test_df, 4)


if __name__ == '__main__':
  absltest.main()
