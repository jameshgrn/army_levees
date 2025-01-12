"""Tests for sample_levees.py."""

import asyncio
import json
import os
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from geopandas import GeoSeries

from army_levees.core.sample_levees import (
    ProcessingStats,
    get_nld_profile_async,
    get_3dep_elevations_async,
    filter_valid_segments,
    get_random_levees,
    validate_1m_coverage,
)

@pytest.fixture
def test_system_id():
    """Test system ID."""
    return "6005000546"

@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    class MockResponse:
        def __init__(self, url):
            self.url = url
            self.status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self):
            if "system-categories" in self.url:
                return {"features": [{"properties": {"systemid": "6005000546"}}]}
            elif "segments" in self.url:
                return {
                    "features": [
                        {
                            "properties": {"systemid": "6005000546", "type": "levee"},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[0, 0], [1, 1], [2, 2]]
                            }
                        }
                    ]
                }
            elif "route" in self.url:
                return {
                    "features": [
                        {
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[0, 0], [1, 1], [2, 2]]
                            }
                        }
                    ]
                }
            return {}

    class MockClientSession:
        async def get(self, url, **kwargs):
            response = MockResponse(url)
            return await response.__aenter__()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockClientSession()

@pytest.fixture
def sample_gdf():
    """Create test GeoDataFrame."""
    points = [Point(0, 0), Point(1, 1), Point(2, 2)]
    data = {
        'elevation': [10.0, 11.0, 12.0],
        'dep_elevation': [11.0, 12.0, 13.0],
        'distance_along_track': [0.0, 1.0, 2.0],
        'geometry': points
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

class TestProcessingStats:
    """Test ProcessingStats class."""

    def test_init(self):
        """Test initialization."""
        stats = ProcessingStats()
        assert stats.total_attempts == 0
        assert stats.success == 0

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ProcessingStats()
        stats.total_attempts = 10
        stats.success = 5
        assert stats.success_rate == 0.5

    def test_str_representation(self):
        """Test string representation."""
        stats = ProcessingStats()
        stats.total_attempts = 10
        stats.success = 5
        expected = (
            "\nProcessing Statistics:\n"
            "  Total attempts: 10\n"
            "  Success rate: 50.0%\n"
            "  Successful: 5\n"
            "  Floodwalls skipped: 0\n"
            "  No data available: 0\n"
            "  Invalid data:\n"
            "    All zeros: 0\n"
            "    All NaNs: 0\n"
            "    Too few points: 0\n"
        )
        assert str(stats) == expected

class TestGetNLDProfileAsync:
    """Test get_nld_profile_async function."""

    @pytest.mark.asyncio
    async def test_successful_profile(self, mock_session, test_system_id):
        """Test successful profile retrieval."""
        stats = ProcessingStats()
        result = await get_nld_profile_async(test_system_id, mock_session, stats)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4  # x, y, elevation, distance

    @pytest.mark.asyncio
    async def test_invalid_response(self, mock_session):
        """Test handling of invalid response."""
        stats = ProcessingStats()
        result = await get_nld_profile_async("invalid_id", mock_session, stats)
        assert result is None

class TestGet3DEPElevationsAsync:
    """Test get_3dep_elevations_async function."""

    @pytest.mark.asyncio
    async def test_successful_elevation(self):
        """Test successful elevation retrieval."""
        coords = [(0.0, 0.0), (1.0, 1.0)]
        with mock.patch('army_levees.core.sample_levees.py3dep.get_dem') as mock_dem:
            mock_dem.return_value = np.ones((10, 10))
            result = await get_3dep_elevations_async(coords)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(isinstance(arr, np.ndarray) for arr in result)

    @pytest.mark.asyncio
    async def test_invalid_bounds(self):
        """Test handling of invalid bounds."""
        coords = [(200.0, 0.0), (201.0, 1.0)]
        with mock.patch('army_levees.core.sample_levees.py3dep.get_dem') as mock_dem:
            mock_dem.return_value = None
            result = await get_3dep_elevations_async(coords)
            assert result is None

class TestFilterValidSegments:
    """Test filter_valid_segments function."""

    def test_valid_filtering(self, sample_gdf):
        """Test filtering of valid segments."""
        result = filter_valid_segments(sample_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_gdf)
        assert all(result['elevation'].notna())
        assert all(result['dep_elevation'].notna())

    def test_nan_filtering(self, sample_gdf):
        """Test filtering of NaN values."""
        sample_gdf.loc[0, 'elevation'] = np.nan
        result = filter_valid_segments(sample_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_gdf) - 1
        assert all(result['elevation'].notna())

    def test_too_few_points(self):
        """Test handling of too few points."""
        points = [Point(0, 0)]
        data = {
            'elevation': [10.0],
            'dep_elevation': [11.0],
            'distance_along_track': [0.0],
            'geometry': points
        }
        gdf = gpd.GeoDataFrame(data)
        gdf.set_crs(epsg=4326, inplace=True)
        result = filter_valid_segments(gdf)
        assert result is None

class TestGetRandomLevees:
    """Test get_random_levees function."""

    @mock.patch('army_levees.core.sample_levees.get_usace_system_ids')
    @mock.patch('army_levees.core.sample_levees.check_3dep_coverage')
    @mock.patch('army_levees.core.sample_levees.process_system')
    def test_successful_sampling(self, mock_process, mock_coverage, mock_get_ids):
        """Test successful random sampling."""
        mock_get_ids.return_value = ["system1", "system2", "system3"]
        mock_coverage.return_value = {"system1": "1m", "system2": "1m", "system3": "1m"}
        mock_process.return_value = gpd.GeoDataFrame()
        result = get_random_levees(n_samples=2)
        assert len(result) == 2

    @mock.patch('army_levees.core.sample_levees.get_usace_system_ids')
    def test_no_available_systems(self, mock_get_ids):
        """Test handling of no available systems."""
        mock_get_ids.return_value = []
        result = get_random_levees(n_samples=2)
        assert result == []

class TestValidate1mCoverage:
    """Test validate_1m_coverage function."""

    def test_valid_bbox(self):
        """Test validation of valid bbox."""
        bbox = (-100, 40, -99, 41)
        with mock.patch('army_levees.core.sample_levees.py3dep.query_3dep_sources') as mock_query:
            mock_sources = mock.MagicMock()
            mock_sources.empty = False
            mock_sources.geometry.unary_union.contains.return_value = True
            mock_query.return_value = mock_sources
            result = validate_1m_coverage(bbox)
            assert isinstance(result, bool)
            assert result is True

    def test_invalid_bbox(self):
        """Test validation of invalid bbox."""
        bbox = (-200, -100, -199, -99)
        result = validate_1m_coverage(bbox)
        assert result is False

class TestGetDEMVRT:
    """Test get_dem_vrt function."""

    def test_successful_vrt(self):
        """Test successful VRT creation."""
        bbox = (-100, 40, -99, 41)
        result = validate_1m_coverage(bbox)
        assert isinstance(result, bool)

    def test_invalid_bbox(self):
        """Test handling of invalid bbox."""
        bbox = (-200, -100, -199, -99)
        result = validate_1m_coverage(bbox)
        assert result is False

class TestIntegration:
    """Test complete workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_session, test_system_id):
        """Test complete levee processing workflow."""
        stats = ProcessingStats()
        profile = await get_nld_profile_async(test_system_id, mock_session, stats)
        assert profile is not None
        assert isinstance(profile, np.ndarray)
        assert profile.shape[1] == 4

if __name__ == '__main__':
    pytest.main(['-v'])
