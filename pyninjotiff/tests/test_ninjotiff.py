#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test the ninjotiff writing."""

import numpy as np
import datetime
import tempfile
import xarray as xr
import dask.array as da
import colorsys
import pytest

TIME = datetime.datetime.utcnow()
DELETE_FILES = True


class FakeImage(object):
    """Fake Image object for testing purposes."""

    def __init__(self, data):
        """Initialize the image."""
        self.mode = ''.join(data.bands.values)
        self.data = data

    def finalize(self, fill_value=None, dtype=None):
        if dtype is None:
            dtype = np.uint8
        if np.issubdtype(self.data.dtype, np.floating) and np.issubdtype(dtype, np.integer):
            res = self.data.clip(0, 1) * np.iinfo(dtype).max
            res = res.astype(dtype)
        else:
            res = self.data
        return [res.astype(dtype)]


class FakeArea(object):
    def __init__(self, proj_dict, extent, y_size, x_size):
        self.proj_dict = proj_dict
        self.area_extent = extent
        self.x_size, self.y_size = x_size, y_size
        self.pixel_size_x = (extent[2] - extent[0]) / x_size
        self.pixel_size_y = (extent[3] - extent[1]) / y_size


def test_write_bw():
    """Test saving a BW image."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)
    scale = 1.0 / 120
    offset = 0.0
    attrs = dict([('resolution', 1050),
                  ('polarization', None),
                  ('platform_name', 'NOAA-18'),
                  ('sensor', 'avhrr-3'),
                  ('units', '%'),
                  ('name', '1'),
                  ('level', None),
                  ('modifiers', ()),
                  ('wavelength', (10.3, 10.8, 11.3)),
                  ('calibration', 'brightness_temperature'),
                  ('start_time', TIME - datetime.timedelta(minutes=5)),
                  ('end_time', TIME),
                  ('area', area),
                  ('ancillary_variables', []),
                  ('enhancement_history', [{'offset': offset, 'scale': scale}])])

    kwargs = {'ch_min_measurement_unit': np.array([0]),
              'ch_max_measurement_unit': np.array([120]),
              'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 100015, 'data_cat': 'PORN', 'data_source': 'SMHI',
              'physic_unit': '%', 'nbits': 8}

    data = da.tile(da.repeat(da.arange(4, chunks=1024) /
                             3.0, 256), 1024).reshape((1, 1024, 1024))
    data = xr.DataArray(data, coords={'bands': ['L']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)
    img = FakeImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        assert(np.allclose(res[0, 0, ::256],
                           np.array([256, 22016, 43520, 65280])))


def test_write_bw_inverted_ir():
    """Test saving a BW image."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)
    scale = 1.0 / 120
    offset = 70.0 / 120
    attrs = dict([('resolution', 1050),
                  ('polarization', None),
                  ('platform_name', 'NOAA-18'),
                  ('sensor', 'avhrr-3'),
                  ('units', 'K'),
                  ('name', '4'),
                  ('level', None),
                  ('modifiers', ()),
                  ('wavelength', (10.3, 10.8, 11.3)),
                  ('calibration', 'brightness_temperature'),
                  ('start_time', TIME - datetime.timedelta(minutes=15)),
                  ('end_time', TIME - datetime.timedelta(minutes=10)),
                  ('area', area),
                  ('ancillary_variables', []),
                  ('enhancement_history', [{'offset': offset, 'scale': scale}])])

    kwargs = {'ch_min_measurement_unit': np.array([-70]),
              'ch_max_measurement_unit': np.array([50]),
              'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 900015, 'data_cat': 'PORN', 'data_source': 'SMHI',
              'physic_unit': 'C', 'nbits': 8}

    data = da.tile(da.repeat(da.arange(4, chunks=1024) /
                             3.0, 256), 1024).reshape((1, 1024, 1024))
    data = xr.DataArray(data, coords={'bands': ['L']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)
    img = FakeImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        assert(np.allclose(res[0, 0, ::256],
                           np.array([65024, 43264, 21760, 0])))


def test_write_bw_fill():
    """Test saving a BW image with transparency."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)
    scale = 1.0 / 120
    offset = 0.0
    attrs = dict([('resolution', 1050),
                  ('polarization', None),
                  ('platform_name', 'NOAA-18'),
                  ('sensor', 'avhrr-3'),
                  ('units', '%'),
                  ('name', '1'),
                  ('level', None),
                  ('modifiers', ()),
                  ('wavelength', (10.3, 10.8, 11.3)),
                  ('calibration', 'brightness_temperature'),
                  ('start_time', TIME - datetime.timedelta(minutes=25)),
                  ('end_time', TIME - datetime.timedelta(minutes=20)),
                  ('area', area),
                  ('ancillary_variables', []),
                  ('enhancement_history', [{'offset': offset, 'scale': scale}])])

    kwargs = {'ch_min_measurement_unit': np.array([0]),
              'ch_max_measurement_unit': np.array([120]),
              'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 100015, 'data_cat': 'PORN', 'data_source': 'SMHI',
              'physic_unit': '%', 'nbits': 8}

    data1 = da.tile(da.repeat(da.arange(4, chunks=1024) /
                              3.0, 256), 256).reshape((1, 256, 1024))
    datanan = da.ones((1, 256, 1024), chunks=1024) * np.nan
    data2 = da.tile(da.repeat(da.arange(4, chunks=1024) /
                              3.0, 256), 512).reshape((1, 512, 1024))
    data = da.concatenate((data1, datanan, data2), axis=1)
    data = xr.DataArray(data, coords={'bands': ['L']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)
    img = FakeImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        assert(np.allclose(res[0, 0, ::256],
                           np.array([256, 22016, 43520, 65280])))


def test_write_bw_inverted_ir_fill():
    """Test saving a BW image with transparency."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)
    scale = 1.0 / 120
    offset = 70.0 / 120
    attrs = dict([('resolution', 1050),
                  ('polarization', None),
                  ('platform_name', 'NOAA-18'),
                  ('sensor', 'avhrr-3'),
                  ('units', 'K'),
                  ('name', '4'),
                  ('level', None),
                  ('modifiers', ()),
                  ('wavelength', (10.3, 10.8, 11.3)),
                  ('calibration', 'brightness_temperature'),
                  ('start_time', TIME - datetime.timedelta(minutes=35)),
                  ('end_time', TIME - datetime.timedelta(minutes=30)),
                  ('area', area),
                  ('ancillary_variables', []),
                  ('enhancement_history', [{'offset': offset, 'scale': scale}])])

    kwargs = {'ch_min_measurement_unit': np.array([-70]),
              'ch_max_measurement_unit': np.array([50]),
              'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 900015, 'data_cat': 'PORN', 'data_source': 'SMHI',
              'physic_unit': 'C', 'nbits': 8}

    data1 = da.tile(da.repeat(da.arange(4, chunks=1024) /
                              3.0, 256), 256).reshape((1, 256, 1024))
    datanan = da.ones((1, 256, 1024), chunks=1024) * np.nan
    data2 = da.tile(da.repeat(da.arange(4, chunks=1024) /
                              3.0, 256), 512).reshape((1, 512, 1024))
    data = da.concatenate((data1, datanan, data2), axis=1)
    data = xr.DataArray(data, coords={'bands': ['L']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)
    img = FakeImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        assert(np.allclose(res[0, 0, ::256],
                           np.array([65024, 43264, 21760, 0])))


def test_write_rgb():
    """Test saving a non-trasparent RGB."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)

    x_size, y_size = 1024, 1024
    arr = np.zeros((3, y_size, x_size))
    radius = min(x_size, y_size) / 2.0
    centre = x_size / 2, y_size / 2

    for x in range(x_size):
        for y in range(y_size):
            rx = x - centre[0]
            ry = y - centre[1]
            s = ((x - centre[0])**2.0 + (y - centre[1])**2.0)**0.5 / radius
            if s <= 1.0:
                h = ((np.arctan2(ry, rx) / np.pi) + 1.0) / 2.0
                rgb = colorsys.hsv_to_rgb(h, s, 1.0)
                arr[:, y, x] = np.array(rgb)

    attrs = dict([('platform_name', 'NOAA-18'),
                  ('resolution', 1050),
                  ('polarization', None),
                  ('level', None),
                  ('sensor', 'avhrr-3'),
                  ('ancillary_variables', []),
                  ('area', area),
                  ('start_time', TIME - datetime.timedelta(minutes=45)),
                  ('end_time', TIME - datetime.timedelta(minutes=40)),
                  ('wavelength', None),
                  ('optional_datasets', []),
                  ('standard_name', 'overview'),
                  ('name', 'overview'),
                  ('prerequisites', [0.6, 0.8, 10.8]),
                  ('optional_prerequisites', []),
                  ('calibration', None),
                  ('modifiers', None),
                  ('mode', 'RGB'),
                  ('enhancement_history', [{'scale': np.array([1,  1, -1]), 'offset': np.array([0, 0, 1])},
                                           {'scale': np.array([0.0266347, 0.03559078, 0.01329783]),
                                            'offset': np.array([-0.02524969, -0.01996642,  3.8918446])},
                                           {'gamma': 1.6}])])

    kwargs = {'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 6500015, 'data_cat': 'PPRN', 'data_source': 'SMHI', 'nbits': 8}
    data = da.from_array(arr.clip(0, 1), chunks=1024)
    data = xr.DataArray(data, coords={'bands': ['R', 'G', 'B']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)

    from trollimage.xrimage import XRImage
    img = XRImage(data)

    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=False, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        for idx in range(3):
            np.testing.assert_allclose(res[:, :, idx], np.round(
                arr[idx, :, :] * 255).astype(np.uint8))


def test_write_rgb_with_a():
    """Test saving a transparent RGB."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)

    x_size, y_size = 1024, 1024
    arr = np.zeros((3, y_size, x_size))
    radius = min(x_size, y_size) / 2.0
    centre = x_size / 2, y_size / 2

    for x in range(x_size):
        for y in range(y_size):
            rx = x - centre[0]
            ry = y - centre[1]
            s = ((x - centre[0])**2.0 + (y - centre[1])**2.0)**0.5 / radius
            if s <= 1.0:
                h = ((np.arctan2(ry, rx) / np.pi) + 1.0) / 2.0
                rgb = colorsys.hsv_to_rgb(h, s, 1.0)
                arr[:, y, x] = np.array(rgb)
            else:
                arr[:, y, x] = np.nan

    attrs = dict([('platform_name', 'NOAA-18'),
                  ('resolution', 1050),
                  ('polarization', None),
                  ('start_time', TIME - datetime.timedelta(minutes=55)),
                  ('end_time', TIME - datetime.timedelta(minutes=50)),
                  ('level', None),
                  ('sensor', 'avhrr-3'),
                  ('ancillary_variables', []),
                  ('area', area),
                  ('wavelength', None),
                  ('optional_datasets', []),
                  ('standard_name', 'overview'),
                  ('name', 'overview'),
                  ('prerequisites', [0.6, 0.8, 10.8]),
                  ('optional_prerequisites', []),
                  ('calibration', None),
                  ('modifiers', None),
                  ('mode', 'RGB'),
                  ('enhancement_history', [{'scale': np.array([1,  1, -1]), 'offset': np.array([0, 0, 1])},
                                           {'scale': np.array([0.0266347, 0.03559078, 0.01329783]),
                                            'offset': np.array([-0.02524969, -0.01996642,  3.8918446])},
                                           {'gamma': 1.6}])])

    kwargs = {'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 6500015, 'data_cat': 'PPRN', 'data_source': 'SMHI', 'nbits': 8}
    data = da.from_array(arr.clip(0, 1), chunks=1024)

    data = xr.DataArray(data, coords={'bands': ['R', 'G', 'B']}, dims=[
                        'bands', 'y', 'x'], attrs=attrs)
    from trollimage.xrimage import XRImage
    img = XRImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        for idx in range(3):
            np.testing.assert_allclose(res[:, :, idx], np.round(
                np.nan_to_num(arr[idx, :, :]) * 255).astype(np.uint8))
        np.testing.assert_allclose(res[:, :, 3] == 0, np.isnan(arr[0, :, :]))


@pytest.mark.skip(reason="this is no implemented yet.")
def test_write_rgb_classified():
    """Test saving a transparent RGB."""
    from pyninjotiff.ninjotiff import save
    from pyninjotiff.tifffile import TiffFile

    area = FakeArea({'ellps': 'WGS84', 'lat_0': 90.0, 'lat_ts': 60.0, 'lon_0': 0.0, 'proj': 'stere'},
                    (-1000000.0, -4500000.0, 2072000.0, -1428000.0),
                    1024, 1024)

    x_size, y_size = 1024, 1024
    arr = np.zeros((3, y_size, x_size))

    attrs = dict([('platform_name', 'NOAA-18'),
                  ('resolution', 1050),
                  ('polarization', None),
                  ('start_time', TIME - datetime.timedelta(minutes=55)),
                  ('end_time', TIME - datetime.timedelta(minutes=50)),
                  ('level', None),
                  ('sensor', 'avhrr-3'),
                  ('ancillary_variables', []),
                  ('area', area),
                  ('wavelength', None),
                  ('optional_datasets', []),
                  ('standard_name', 'overview'),
                  ('name', 'overview'),
                  ('prerequisites', [0.6, 0.8, 10.8]),
                  ('optional_prerequisites', []),
                  ('calibration', None),
                  ('modifiers', None),
                  ('mode', 'P')])

    kwargs = {'compute': True, 'fill_value': None, 'sat_id': 6300014,
              'chan_id': 1700015, 'data_cat': 'PPRN', 'data_source': 'SMHI', 'nbits': 8}

    data1 = da.tile(da.repeat(da.arange(4, chunks=1024), 256), 256).reshape((1, 256, 1024))
    datanan = da.ones((1, 256, 1024), chunks=1024) * 4
    data2 = da.tile(da.repeat(da.arange(4, chunks=1024), 256), 512).reshape((1, 512, 1024))
    data = da.concatenate((data1, datanan, data2), axis=1)
    data = xr.DataArray(data, coords={'bands': ['P']}, dims=['bands', 'y', 'x'], attrs=attrs)

    from trollimage.xrimage import XRImage
    img = XRImage(data)
    with tempfile.NamedTemporaryFile(delete=DELETE_FILES) as tmpfile:
        filename = tmpfile.name
        if not DELETE_FILES:
            print(filename)
        save(img, filename, data_is_scaled_01=True, **kwargs)
        tif = TiffFile(filename)
        res = tif[0].asarray()
        for idx in range(3):
            np.testing.assert_allclose(res[:, :, idx], np.round(
                np.nan_to_num(arr[idx, :, :]) * 255).astype(np.uint8))
        np.testing.assert_allclose(res[:, :, 3] == 0, np.isnan(arr[0, :, :]))
