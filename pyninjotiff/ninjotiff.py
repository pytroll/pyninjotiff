# -*- coding: utf-8 -*-
# ninjotiff.py
#
# Created on Mon Apr 15 13:41:55 2013
#
# A big amount of the tiff writer are (PFE) from
# https://github.com/davidh-ssec/polar2grid by David Hoese
#
# License:
# Copyright (C) 2013 Space Science and Engineering Center (SSEC),
#  University of Wisconsin-Madison.
#  Lars Ørum Rasmussen, DMI.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Original scripts and automation included as part of this package are
# distributed under the GNU GENERAL PUBLIC LICENSE agreement version 3.
# Binary executable files included as part of this software package are
# copyrighted and licensed by their respective organizations, and
# distributed consistent with their licensing terms.
#
# Edited by Christian Kliche (Ernst Basler + Partner) to replace pylibtiff with
# a modified version of tifffile.py (created by Christoph Gohlke)
"""Ninjotiff writing utility."""

import calendar
import logging
import os
from copy import deepcopy
from datetime import datetime

import numpy as np

from pyproj import Proj
from pyresample.utils import proj4_radius_parameters

from pyninjotiff import tifffile as local_tifffile

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
#
# Ninjo tiff tags from DWD
#
# -------------------------------------------------------------------------
# Geotiff tags.
GTF_ModelPixelScale = 33550
GTF_ModelTiepoint = 33922

# Ninjo tiff tags
NINJO_TAGS = {
    "NTD_Magic": 40000,
    "NTD_SatelliteNameID": 40001,
    "NTD_DateID": 40002,
    "NTD_CreationDateID": 40003,
    "NTD_ChannelID": 40004,
    "NTD_HeaderVersion": 40005,
    "NTD_FileName": 40006,
    "NTD_DataType": 40007,
    "NTD_SatelliteNumber": 40008,
    "NTD_ColorDepth": 40009,
    "NTD_DataSource": 40010,
    "NTD_XMinimum": 40011,
    "NTD_XMaximum": 40012,
    "NTD_YMinimum": 40013,
    "NTD_YMaximum": 40014,
    "NTD_Projection": 40015,
    "NTD_MeridianWest": 40016,
    "NTD_MeridianEast": 40017,
    "NTD_EarthRadiusLarge": 40018,
    "NTD_EarthRadiusSmall": 40019,
    "NTD_GeodeticDate": 40020,
    "NTD_ReferenceLatitude1": 40021,
    "NTD_ReferenceLatitude2": 40022,
    "NTD_CentralMeridian": 40023,
    "NTD_PhysicValue": 40024,
    "NTD_PhysicUnit": 40025,
    "NTD_MinGrayValue": 40026,
    "NTD_MaxGrayValue": 40027,
    "NTD_Gradient": 40028,
    "NTD_AxisIntercept": 40029,
    "NTD_ColorTable": 40030,
    "NTD_Description": 40031,
    "NTD_OverflightDirection": 40032,
    "NTD_GeoLatitude": 40033,
    "NTD_GeoLongitude": 40034,
    "NTD_Altitude": 40035,
    "NTD_AOSAsimuth": 40036,
    "NTD_LOSAsimuth": 40037,
    "NTD_MaxElevation": 40038,
    "NTD_OverflightTime": 40039,
    "NTD_IsBlackLineCorrection": 40040,
    "NTD_IsAtmosphereCorrected": 40041,
    "NTD_IsCalibrated": 40042,
    "NTD_IsNormalized": 40043,
    "NTD_OriginalHeader": 40044,
    "NTD_IsValueTableAvailable": 40045,
    "NTD_ValueTableStringField": 40046,
    "NTD_ValueTableFloatField": 40047,
    "NTD_TransparentPixel": 50000
}

NINJO_TAGS_INV = dict((v, k) for k, v in NINJO_TAGS.items())

#
# model_pixel_scale_tag_count ? ...
# Sometimes DWD product defines an array of length 2 (instead of 3 (as in geotiff)).
#
MODEL_PIXEL_SCALE_COUNT = int(os.environ.get(
    "GEOTIFF_MODEL_PIXEL_SCALE_COUNT", 3))


# -------------------------------------------------------------------------
#
# Read Ninjo products config file.
#
# -------------------------------------------------------------------------
def get_writer_config(config_fname, prod, single_product_config, scn_metadata):
    """Writer_config function for Trollflow_sat: calls the get_product_config function.

    :Parameters:
       config_fname: str
            Name of the Ninjo product configuration file

       prod: str
            Name of Ninjo product.

       single_product_config: dict
            config params for the current product

       scn_metadata: dict
            Satpy satellite data
    """
    ninjo_product = prod
    if 'ninjo_product_name' in single_product_config:
        ninjo_product = single_product_config['ninjo_product_name']
    return get_product_config(ninjo_product, True, config_fname)


def get_product_config(product_name, force_read=False, config_filename=None):
    """Read Ninjo configuration entry for a given product name.

    :Parameters:
        product_name : str
            Name of Ninjo product.

    :Arguments:
        force_read : Boolean
            Force re-reading config file.

    **Notes**:
        * It will look for a *ninjotiff_products.cfg* in MPOP's
          configuration directory defined by *PPP_CONFIG_DIR*.
        * As an example, see *ninjotiff_products.cfg.template* in
          MPOP's *etc* directory.
    """
    return ProductConfigs()(product_name, force_read, config_filename)


class _Singleton(type):

    def __init__(self, name_, bases_, dict_):
        """Init the singleton."""
        super(_Singleton, self).__init__(name_, bases_, dict_)
        self.instance = None

    def __call__(self, *args, **kwargs):
        """Call the singleton."""
        if self.instance is None:
            self.instance = super(_Singleton, self).__call__(*args, **kwargs)
        return self.instance


class ProductConfigs(object):
    """Product config."""

    __metaclass__ = _Singleton  # noqa

    def __init__(self):
        """Init the product config."""
        self.read_config()

    def __call__(self, product_name, force_read=False, config_filename=None):
        """Call the product config."""
        if force_read:
            self.read_config(config_filename)
        if product_name in self._products:
            return self._products[product_name]
        else:
            return {}

    @property
    def product_names(self):
        """Get the product names."""
        return sorted(self._products.keys())

    def read_config(self, config_filename=None):
        """Read the ninjo products config file."""
        from six.moves.configparser import RawConfigParser
        import ast

        def _eval(val):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return str(val)

        if config_filename is not None:
            filename = self._find_a_config_file(config_filename)
        else:
            filename = self._find_a_config_file('ninjotiff_products.cfg')
        log.info("Reading Ninjo config file: '%s'" % filename)

        cfg = RawConfigParser()
        products = {}
        if filename is not None:
            cfg.read(filename)
            for sec in cfg.sections():
                prd = {}
                for key, val in cfg.items(sec):
                    prd[key] = _eval(val)
                products[sec] = prd
        self._products = products

    @staticmethod
    def _find_a_config_file(fname):
        # if config file (fname) is not found as absolute path: look for the
        # config file in the PPP_CONFIG_DIR or current dir
        name_ = os.path.abspath(os.path.expanduser(fname))
        if os.path.isfile(name_):
            return name_
        else:
            home_ = os.path.dirname(os.path.abspath(__file__))
            penv_ = os.environ.get('PPP_CONFIG_DIR', '')
            for fname_ in [os.path.join(x, name_) for x in (home_, penv_)]:
                if os.path.isfile(fname_):
                    return fname_
        # raise ValueError("Could not find a Ninjo tiff config file")


# -------------------------------------------------------------------------
#
# Write Ninjo Products
#
# -------------------------------------------------------------------------
def _get_physic_value(physic_unit):
    # return Ninjo's physics unit and value.
    if physic_unit.upper() in ('K', 'KELVIN'):
        return 'Kelvin', 'T'
    elif physic_unit.upper() in ('C', 'CELSIUS'):
        return 'Celsius', 'T'
    elif physic_unit == '%':
        return physic_unit, 'Reflectance'
    elif physic_unit.upper() in ('MW M-2 SR-1 (CM-1)-1',):
        return physic_unit, 'Radiance'
    else:
        return physic_unit, 'Unknown'


def _get_projection_name(area_def):
    # return Ninjo's projection name.
    proj_name = area_def.proj_dict['proj']
    if proj_name in ('eqc',):
        return 'PLAT'
    elif proj_name in ('merc',):
        return 'MERC'
    elif proj_name in ('stere',):
        lat_0 = area_def.proj_dict['lat_0']
        if lat_0 < 0:
            return 'SPOL'
        else:
            return 'NPOL'
    # FIXME: this feels like a hack
    return area_def.proj_id.split('_')[-1]


def _get_pixel_size(projection_name, area_def):
    if projection_name in ['PLAT', 'MERC', ]:
        proj = Proj(area_def.proj_dict)
        upper_left = proj(
            area_def.area_extent[0],
            area_def.area_extent[3],
            inverse=True)
        lower_right = proj(
            area_def.area_extent[2],
            area_def.area_extent[1],
            inverse=True)
        pixel_size = abs(lower_right[0] - upper_left[0]) / area_def.shape[1],\
            abs(upper_left[1] - lower_right[1]) / area_def.shape[0]

    elif projection_name in ('NPOL', 'SPOL',):
        a, b = proj4_radius_parameters(area_def.proj_dict)
        pixel_size = (np.rad2deg(area_def.pixel_size_x / a),
                      np.rad2deg(area_def.pixel_size_y / b))
    else:
        raise ValueError(
            "Could not determine pixel size from projection name '%s'" %
            projection_name + " (Unknown)")
    return pixel_size


def _get_satellite_altitude(filename):
    # Guess altitude (probably no big deal if we fail).
    sat_altitudes = {'MSG': 36000.0,
                     'METOP': 817.0,
                     'NOAA': 870.0}

    filename = os.path.basename(filename).upper()
    for nam_, alt_ in sat_altitudes.items():
        if nam_ in filename:
            return alt_
    return None


def _finalize(img, dtype=np.uint8, value_range_measurement_unit=None,
              data_is_scaled_01=True, fill_value=None):
    """Finalize a trollimage.Image for Ninjo.

    Specialy take care of phycical scale and offset.

    :Parameters:
        img : mpop.imageo.img.GeoImage
            See MPOP's documentation.
        dtype : bits per sample np.uint8 or np.uint16 (default: np.uint8)
        value_range_measurement_unit: list or tuple
            Defining minimum and maximum value range. Data will be clipped into
            that range. Default is no clipping and auto scale.
        data_is_scaled_01: boolean
            If true (default), input data is assumed to be in the [0.0, 1.0] range.

    :Returns:
        image : numpy.array
            Final image.
        scale : float
            Scale for transform pixel value to physical value.
        offset : float
            Offset for transform pixel value to physical value.
        fill_value : int
            Value for masked out pixels.

    **Notes**:
        physic_val = image*scale + offset
        Example values for value_range_measurement_unit are (0, 125) or (40.0, -87.5)

    ***Warning***
        Only the 'L' and 'RGB' cases are compatible with xarray.XRImage.
        They still have to  be tested thoroughly.
    """
    if img.mode == 'L':
        # PFE: mpop.satout.cfscene
        if isinstance(img, np.ma.MaskedArray):
            data = img.channels[0]
        else:
            # TODO: check what is the correct fill value for NinJo!
            if fill_value is not None:
                log.debug("Forcing fill value to %s", fill_value)
            # Go back to the masked_array for compatibility
            # with the following part of the code.
            if (np.issubdtype(img.data[0].dtype, np.integer)
                    and '_FillValue' in img.data[0].attrs):
                nodata_value = img.data[0].attrs['_FillValue']
                data = img.data[0].values
                data = np.ma.array(data, mask=(data == nodata_value))
            else:
                data = img.data[0].to_masked_array()

        fill_value = fill_value if fill_value is not None else np.iinfo(dtype).min

        log.debug("Before scaling: %.2f, %.2f, %.2f" %
                  (data.min(), data.mean(), data.max()))

        if np.ma.count_masked(data) == data.size:
            # All data is masked
            data = np.ones(data.shape, dtype=dtype) * fill_value
            scale = 1
            offset = 0
        else:
            if value_range_measurement_unit and data_is_scaled_01:
                # No additional scaling of the input data - assume that data is
                # within [0.0, 1.0] and interpret 0.0 as
                # value_range_measurement_unit[0] and 1.0 as
                # value_range_measurement_unit[1]

                # Make room for the transparent pixel value.
                data = data.clip(0, 1)
                data *= (np.iinfo(dtype).max - 1)
                data += 1

                scale = ((value_range_measurement_unit[1]
                          - value_range_measurement_unit[0])
                         / (np.iinfo(dtype).max - 1))
                # Handle the case where all data has the same value.
                scale = scale or 1
                offset = value_range_measurement_unit[0]

                mask = data.mask
                data = np.round(data.data).astype(dtype)
                offset -= scale

                if fill_value is None:
                    fill_value = 0

            else:
                if value_range_measurement_unit:
                    data.clip(value_range_measurement_unit[0],
                              value_range_measurement_unit[1], data)
                    chn_min = value_range_measurement_unit[0]
                    chn_max = value_range_measurement_unit[1]
                    log.debug("Scaling, using value range %.2f - %.2f" %
                              (value_range_measurement_unit[0], value_range_measurement_unit[1]))
                else:
                    chn_max = data.max()
                    chn_min = data.min()
                    log.debug("Doing auto scaling")

                # Make room for transparent pixel.
                scale = ((chn_max - chn_min) /
                         (np.iinfo(dtype).max - 1.0))

                # Handle the case where all data has the same value.
                scale = scale or 1
                offset = chn_min

                # Scale data to dtype, and adjust for transparent pixel forced
                # to be minimum.
                mask = data.mask
                data = 1 + ((data.data - offset) / scale).astype(dtype)
                offset -= scale

            data[mask] = fill_value

            if log.getEffectiveLevel() == logging.DEBUG:
                d__ = np.ma.array(data, mask=(data == fill_value))
                log.debug("After scaling:  %.2f, %.2f, %.2f" % (d__.min(),
                                                                d__.mean(),
                                                                d__.max()))
                d__ = data * scale + offset
                d__ = np.ma.array(d__, mask=(data == fill_value))
                log.debug("Rescaling:      %.2f, %.2f, %.2f" % (d__.min(),
                                                                d__.mean(),
                                                                d__.max()))
                del d__

        return data, scale, offset, fill_value

    elif img.mode == 'RGB':
        if isinstance(img, np.ma.MaskedArray):
            channels, fill_value = img._finalize(dtype)
        else:
            data, mode = img.finalize(fill_value=fill_value, dtype=dtype)
            # Go back to the masked_array for compatibility with
            # the rest of the code.
            channels = data.to_masked_array()
            # Is this fill_value ok or what should it be?
            fill_value = (0, 0, 0, 0)

        if isinstance(img, np.ma.MaskedArray) and fill_value is None:
            mask = (np.ma.getmaskarray(channels[0]) &
                    np.ma.getmaskarray(channels[1]) &
                    np.ma.getmaskarray(channels[2]))
            channels.append((np.ma.logical_not(mask) *
                             np.iinfo(channels[0].dtype).max).astype(channels[0].dtype))
            fill_value = (0, 0, 0, 0)

        data = np.dstack([channel.filled(fill_v)
                          for channel, fill_v in zip(channels, fill_value)])
        return data, 1.0, 0.0, fill_value[0]

    elif img.mode == 'RGBA':
        if not isinstance(img, np.ma.MaskedArray):
            raise NotImplementedError("The 'RGBA' case has not been updated to xarray")
        channels, fill_value = img._finalize(dtype)
        fill_value = fill_value or (0, 0, 0, 0)
        data = np.dstack((channels[0].filled(fill_value[0]),
                          channels[1].filled(fill_value[1]),
                          channels[2].filled(fill_value[2]),
                          channels[3].filled(fill_value[3])))
        return data, 1.0, 0.0, fill_value[0]

    elif img.mode == 'P':
        if not isinstance(img, np.ma.MaskedArray):
            raise NotImplementedError("The 'P' case has not been updated to xarray")
        fill_value = 0
        data = img.channels[0]
        if isinstance(data, np.ma.core.MaskedArray):
            data = data.filled(fill_value)
        data = data.astype(dtype)
        log.debug("Value range: %.2f, %.2f, %.2f" %
                  (data.min(), data.mean(), data.max()))
        return data, 1.0, 0.0, fill_value

    else:
        raise ValueError("Don't know how to handle image mode '%s'" %
                         str(img.mode))


def save(img, filename, ninjo_product_name=None, writer_options=None, data_is_scaled_01=True,
         **kwargs):
    """Ninjo TIFF writer.

    :Parameters:
        img : Trollimage Image instance
        geodef: A pyresample AreaDefinition
            Ninjo support only stere, merc and eqc for now
        filename : str
            The name of the TIFF file to be created
    :Keywords:
        ninjo_product_name : str
            Optional index to Ninjo configuration file.
        writer_options : dict
            options dictionary as defined in MPOP interface
            See _write
        kwargs : dict
            See _write

    **Note**:
        * Some arguments are type casted, since they could come from a config file, read as strings.
        * 8 bits grayscale with a colormap (if specified, inverted for IR channels).
        * 16 bits grayscale with no colormap (if specified, MinIsWhite is set for IR).
        * min value will be reserved for transparent color.
        * If possible mpop.imageo.image's standard finalize will be used.
    """
    if writer_options:
        # add writer_options
        kwargs.update(writer_options)
        if 'ninjo_product_name' in writer_options:
            ninjo_product_name = writer_options['ninjo_product_name']

    dtype = np.uint8
    if 'nbits' in kwargs:
        nbits = int(kwargs['nbits'])
        if nbits == 16:
            dtype = np.uint16

    fill_value = None
    if 'fill_value' in kwargs and kwargs['fill_value'] is not None:
        fill_value = int(kwargs['fill_value'])

    try:
        value_range_measurement_unit = (float(kwargs["ch_min_measurement_unit"]),
                                        float(kwargs["ch_max_measurement_unit"]))
    except KeyError:
        value_range_measurement_unit = None

    # In case we are working on a trollimage.xrimage.XRImage,
    # a conversion to the previously used masked_array is needed

    data, scale, offset, fill_value = _finalize(img,
                                                dtype=dtype,
                                                data_is_scaled_01=data_is_scaled_01,
                                                value_range_measurement_unit=value_range_measurement_unit,
                                                fill_value=fill_value,)

    if isinstance(img, np.ma.MaskedArray):
        area_def = img.info['area']
        time_slot = img.info['start_time']
    else:
        area_def = img.data.area
        time_slot = img.data.start_time

    # Some Ninjo tiff names
    kwargs['gradient'] = scale
    kwargs['axis_intercept'] = offset
    kwargs['transparent_pix'] = fill_value
    kwargs['image_dt'] = time_slot
    kwargs['is_calibrated'] = True
    if img.mode == 'P' and 'cmap' not in kwargs:
        r, g, b = zip(*img.palette)
        r = list((np.array(r) * 255).astype(np.uint8))
        g = list((np.array(g) * 255).astype(np.uint8))
        b = list((np.array(b) * 255).astype(np.uint8))
        if len(r) < 256:
            r += [0] * (256 - len(r))
            g += [0] * (256 - len(g))
            b += [0] * (256 - len(b))
        kwargs['cmap'] = r, g, b
    write(data, filename, area_def, ninjo_product_name, **kwargs)


def ninjo_nav_parameters(options, area_def):
    """Fill options with the navigation parameter in Ninjo format."""
    # TODO: add altitude if available
    proj = Proj(area_def.proj_dict)
    upper_left = proj(
        area_def.area_extent[0],
        area_def.area_extent[3],
        inverse=True)
    lower_right = proj(
        area_def.area_extent[2],
        area_def.area_extent[1],
        inverse=True)

    # Ninjo's projection name.
    options.setdefault('projection', _get_projection_name(area_def))

    # Get pixel size
    if 'pixel_xres' not in options or 'pixel_yres' not in options:
        options['pixel_xres'], options['pixel_yres'] = \
            _get_pixel_size(options['projection'], area_def)

    options['meridian_west'] = upper_left[0]
    options['meridian_east'] = lower_right[0]
    if options['projection'].endswith("POL"):
        if 'lat_ts' in area_def.proj_dict:
            options['ref_lat1'] = area_def.proj_dict['lat_ts']
            options['ref_lat2'] = 0
    else:
        if 'lat_0' in area_def.proj_dict:
            options['ref_lat1'] = area_def.proj_dict['lat_0']
            options['ref_lat2'] = 0
    if 'lon_0' in area_def.proj_dict:
        options['central_meridian'] = area_def.proj_dict['lon_0']

    a, b = proj4_radius_parameters(area_def.proj_dict)
    options['radius_a'] = a
    options['radius_b'] = b
    options['origin_lon'] = upper_left[0]
    options['origin_lat'] = upper_left[1]


def write(image_data, output_fn, area_def, product_name=None, **kwargs):
    """Write a Generic Ninjo TIFF.

    If 'product_name' is given, it will load corresponding Ninjo tiff metadata
    from '${PPP_CONFIG_DIR}/ninjotiff.cfg'. Else, all Ninjo tiff metadata should
    be passed by '**kwargs'. A mixture is allowed, where passed arguments
    overwrite config file.

    :Parameters:
        image_data : 2D numpy array
            Satellite image data to be put into the NinJo compatible tiff
        output_fn : str
            The name of the TIFF file to be created
        area_def: pyresample.geometry.AreaDefinition
            Defintion of area
        product_name : str
            Optional index to Ninjo configuration file.

    :Keywords:
        kwargs : dict
            See _write
    """
    if len(image_data.shape) == 3:
        if image_data.shape[2] == 4:
            shape = (area_def.y_size, area_def.x_size, 4)
            log.info("Will generate RGBA product")
        else:
            shape = (area_def.y_size, area_def.x_size, 3)
            log.info("Will generate RGB product")
        write_rgb = True
    else:
        shape = (area_def.y_size, area_def.x_size)
        write_rgb = False
        log.info("Will generate single band product")

    if image_data.shape != shape:
        raise ValueError("Raster shape %s does not correspond to expected shape %s" % (
            str(image_data.shape), str(shape)))

    # Ninjo's physical units and value.
    # If just a physical unit (e.g. 'C') is passed, it will then be
    # translated into Ninjo's unit and value (e.q 'CELCIUS' and 'T').
    physic_unit = kwargs.get('physic_unit', None)
    if physic_unit and not kwargs.get('physic_value', None):
        kwargs['physic_unit'], kwargs['physic_value'] = \
            _get_physic_value(physic_unit)

    if product_name:
        # If ninjo_product_file in kwargs, load ninjo_product_file as config file
        if 'ninjo_product_file' in kwargs:
            options = deepcopy(get_product_config(product_name, True, kwargs['ninjo_product_file']))
        else:
            options = deepcopy(get_product_config(product_name))
    else:
        options = {}

    options.update(kwargs)  # Update/overwrite with passed arguments

    options['min_gray_val'] = image_data.min()
    options['max_gray_val'] = image_data.max()

    ninjo_nav_parameters(options, area_def)

    _write(image_data, output_fn, write_rgb=write_rgb, **options)


# -----------------------------------------------------------------------------
#
# Write tiff file.
#
# -----------------------------------------------------------------------------
def _write(image_data, output_fn, write_rgb=False, **kwargs):
    """Create a NinJo compatible TIFF file.

    Proudly Found Elsewhere (PFE) https://github.com/davidh-ssec/polar2grid
    by David Hoese.

    Also stores the image as tiles on disk
    and creates a multi-resolution/pyramid/overview set of images
    (deresolution: 2,4,8,16).

    :Parameters:
        image_data : 2D or 3D numpy array
            Satellite image data to be put into the NinJo compatible tiff
            An 3D array (HxWx3) is expected for a RGB image.
        filename : str
            The name of the TIFF file to be created

    :Keywords:
        cmap : tuple/list of 3 lists of uint16's
            Individual RGB arrays describing the color value for the
            corresponding data value.  For example, image data with a data
            type of unsigned 8-bit integers have 256 possible values (0-255).
            So each list in cmap will have 256 values ranging from 0 to
            65535 (2**16 - 1). (default linear B&W colormap)
        sat_id : int
            DWD NinJo Satellite ID number
        chan_id : int
            DWD NinJo Satellite Channel ID number
        data_source : str
            String describing where the data came from (SSEC, EUMCAST)
        tile_width : int
            Width of tiles on disk (default 512)
        tile_length : int
            Length of tiles on disk (default 512)
        data_cat : str
            NinJo specific data category
                - data_cat[0] = P (polar) or G (geostat)
                - data_cat[1] = O (original) or P (product)
                - data_cat[2:4] = RN or RB or RA or RN or AN
                                  (Raster, Bufr, ASCII, NIL)

            Example: 'PORN' or 'GORN' or 'GPRN' or 'PPRN'
        pixel_xres : float
            Nadir view pixel resolution in degrees longitude
        pixel_yres : float
            Nadir view pixel resolution in degrees latitude
        origin_lat : float
            Top left corner latitude
        origin_lon : float
            Top left corner longitude
        image_dt : datetime object
            Python datetime object describing the date and time of the image
            data provided in UTC
        projection : str
            NinJo compatible projection name (NPOL,PLAT,etc.)
        meridian_west : float
            Western image border (default 0.0)
        meridian_east : float
            Eastern image border (default 0.0)
        radius_a : float
            Large/equatorial radius of the earth (default <not written>)
        radius_b : float
            Small/polar radius of the earth (default <not written>)
        ref_lat1 : float
            Reference latitude 1 (default <not written>)
        ref_lat2 : float
            Reference latitude 2 (default <not written>)
        central_meridian : float
            Central Meridian (default <not written>)
        physic_value : str
            Physical value type. Examples:
                - Temperature = 'T'
                - Albedo = 'ALBEDO'
        physic_unit : str
            Physical value units. Examples:
                - 'CELSIUS'
                - '%'
        min_gray_val : int
            Minimum gray value (default 0)
        max_gray_val : int
            Maximum gray value (default 255)
        gradient : float
            Gradient/Slope
        axis_intercept : float
            Axis Intercept
        altitude : float
            Altitude of the data provided (default 0.0)
        is_atmo_corrected : bool
            Is the data atmosphere corrected? (True/1 for yes) (default False/0)
        is_calibrated : bool
            Is the data calibrated? (True/1 for yes) (default False/0)
        is_normalized : bool
            Is the data normalized (True/1 for yes) (default False/0)
        description : str
            Description string to be placed in the output TIFF (optional)
        transparent_pix : int
            Transparent pixel value (default -1)
        compression : int
            zlib compression level (default 6)
        inv_def_temperature_cmap : bool (default True)
            Invert the default colormap if physical value type is 'T'
        omit_filename_path : bool (default False)
            Do not store path in NTD_FileName tag

    :Raises:
        KeyError :
            if required keyword is not provided
    """
    def _raise_value_error(text):
        log.error(text)
        raise ValueError(text)

    def _default_colormap(reverse=False, nbits16=False):
        # Basic B&W colormap
        if nbits16:
            if reverse:
                return [[x for x in range(65535, -1, -1)]] * 3
            return [[x for x in range(65536)]] * 3
        else:
            if reverse:
                return [[x * 256 for x in range(255, -1, -1)]] * 3
            return [[x * 256 for x in range(256)]] * 3

    def _eval_or_none(key, eval_func):
        try:
            return eval_func(kwargs[key])
        except KeyError:
            return None

    def _eval_or_default(key, eval_func, default):
        try:
            return eval_func(kwargs[key])
        except KeyError:
            return default

    log.info("Creating output file '%s'" % (output_fn,))

    # Extract keyword arguments
    cmap = kwargs.pop("cmap", None)
    sat_id = int(kwargs.pop("sat_id"))
    chan_id = int(kwargs.pop("chan_id"))
    data_source = str(kwargs.pop("data_source"))
    tile_width = int(kwargs.pop("tile_width", 512))
    tile_length = int(kwargs.pop("tile_length", 512))
    data_cat = str(kwargs.pop("data_cat"))
    pixel_xres = float(kwargs.pop("pixel_xres"))
    pixel_yres = float(kwargs.pop("pixel_yres"))
    origin_lat = float(kwargs.pop("origin_lat"))
    origin_lon = float(kwargs.pop("origin_lon"))
    image_dt = kwargs.pop("image_dt")
    zero_seconds = kwargs.pop("zero_seconds", False)
    projection = str(kwargs.pop("projection"))
    meridian_west = float(kwargs.pop("meridian_west", 0.0))
    meridian_east = float(kwargs.pop("meridian_east", 0.0))
    radius_a = _eval_or_none("radius_a", float)
    radius_b = _eval_or_none("radius_b", float)
    ref_lat1 = _eval_or_none("ref_lat1", float)
    ref_lat2 = _eval_or_none("ref_lat2", float)
    central_meridian = _eval_or_none("central_meridian", float)
    min_gray_val = int(kwargs.pop("min_gray_val", 0))
    max_gray_val = int(kwargs.pop("max_gray_val", 255))
    altitude = _eval_or_none("altitude", float)
    is_blac_corrected = int(bool(kwargs.pop("is_blac_corrected", 0)))
    is_atmo_corrected = int(bool(kwargs.pop("is_atmo_corrected", 0)))
    is_calibrated = int(bool(kwargs.pop("is_calibrated", 0)))
    is_normalized = int(bool(kwargs.pop("is_normalized", 0)))
    inv_def_temperature_cmap = bool(kwargs.pop("inv_def_temperature_cmap", 1))
    omit_filename_path = bool(kwargs.pop("omit_filename_path", 0))
    description = _eval_or_none("description", str)

    physic_value = str(kwargs.pop("physic_value", 'None'))
    physic_unit = str(kwargs.pop("physic_unit", 'None'))
    gradient = float(kwargs.pop("gradient", 1.0))
    axis_intercept = float(kwargs.pop("axis_intercept", 0.0))
    try:
        transparent_pix = int(kwargs.get("transparent_pix", -1))
    except Exception:
        transparent_pix = kwargs.get("transparent_pix")[0]
    finally:
        kwargs.pop("transparent_pix")

    # Keyword checks / verification

    # Handle colormap or not.
    min_is_white = False
    if not write_rgb and not cmap:
        if physic_value == 'T' and inv_def_temperature_cmap:
            reverse = True
        else:
            reverse = False
        if np.iinfo(image_data.dtype).bits == 8:
            # Always generate colormap for 8 bit gray scale.
            cmap = _default_colormap(reverse)
        elif reverse:
            # No colormap for 16 bit gray scale, but for IR, specify white is
            # minimum.
            min_is_white = True

    if cmap and len(cmap) != 3:
        _raise_value_error(
            "Colormap (cmap) must be a list of 3 lists (RGB), not %d" %
            len(cmap))

    if len(data_cat) != 4:
        _raise_value_error("NinJo data type must be 4 characters")
    if data_cat[0] not in ["P", "G"]:
        _raise_value_error(
            "NinJo data type's first character must be 'P' or 'G' not '%s'" %
            data_cat[0])
    if data_cat[1] not in ["O", "P"]:
        _raise_value_error(
            "NinJo data type's second character must be 'O' or 'P' not '%s'" %
            data_cat[1])
    if data_cat[2:4] not in ["RN", "RB", "RA", "BN", "AN"]:
        _raise_value_error(
            "NinJo data type's last 2 characters must be one of %s not '%s'" %
            ("['RN','RB','RA','BN','AN']", data_cat[2:4]))

    if description is not None and len(description) >= 1000:
        log.error("NinJo description must be less than 1000 characters")
        raise ValueError("NinJo description must be less than 1000 characters")

    file_dt = datetime.utcnow()
    file_epoch = calendar.timegm(file_dt.timetuple())
    if zero_seconds:
        log.debug("Applying zero seconds correction")
        image_dt_corr = datetime(image_dt.year, image_dt.month, image_dt.day,
                                 image_dt.hour, image_dt.minute)
    else:
        image_dt_corr = image_dt
    image_epoch = calendar.timegm(image_dt_corr.timetuple())

    compression = _eval_or_default("compression", int, 6)

    def _create_args(image_data, pixel_xres, pixel_yres):
        log.info("creating tags and data for a resolution %dx%d"
                 % image_data.shape[:2])
        args = dict()
        extra_tags = []
        args['extratags'] = extra_tags
        args['software'] = 'tifffile/pytroll'
        args['compress'] = compression

        args['extrasamples_type'] = 2

        # Built ins
        if write_rgb:
            args["photometric"] = 'rgb'
        else:
            if cmap:
                args["photometric"] = 'palette'
                args["colormap"] = [item for sublist in cmap for item in sublist]
            elif min_is_white:
                args["photometric"] = 'miniswhite'
            else:
                args["photometric"] = 'minisblack'

        # planarconfig, samples_per_pixel, orientation, sample_format set by
        # tifffile.py

        args["tile_width"] = tile_width
        args["tile_length"] = tile_length

        # not necessary to set value SMinSampleValue and SMaxsampleValue
        # TIFF6 Spec: The default for SMinSampleValue and SMaxSampleValue is
        # the full range of the data type
        # args["SMinSampleValue"] = 0
        # args["SMaxsampleValue"] = 255

        # NinJo specific tags
        if description is not None:
            extra_tags.append(
                (NINJO_TAGS["NTD_Description"], 's', 0, description, True))

        # Geo tiff tags
        if MODEL_PIXEL_SCALE_COUNT == 3:
            extra_tags.append((GTF_ModelPixelScale,
                               'd', 3, [pixel_xres, pixel_yres, 0.0], True))
        else:
            extra_tags.append((GTF_ModelPixelScale,
                               'd', 2, [pixel_xres, pixel_yres], True))
        extra_tags.append((GTF_ModelTiepoint,
                           'd', 6, [0.0, 0.0, 0.0, origin_lon, origin_lat, 0.0], True))
        extra_tags.append((NINJO_TAGS["NTD_Magic"], 's', 0, "NINJO", True))
        extra_tags.append(
            (NINJO_TAGS["NTD_SatelliteNameID"], 'I', 1, sat_id, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_DateID"], 'I', 1, image_epoch, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_CreationDateID"], 'I', 1, file_epoch, True))
        extra_tags.append((NINJO_TAGS["NTD_ChannelID"], 'I', 1, chan_id, True))
        extra_tags.append((NINJO_TAGS["NTD_HeaderVersion"], 'i', 1, 2, True))
        if omit_filename_path:
            extra_tags.append((NINJO_TAGS["NTD_FileName"], 's', 0,
                               os.path.basename(output_fn), True))
        else:
            extra_tags.append(
                (NINJO_TAGS["NTD_FileName"], 's', 0, output_fn, True))
        extra_tags.append((NINJO_TAGS["NTD_DataType"], 's', 0, data_cat, True))
        # Hardcoded to 0
        extra_tags.append(
            (NINJO_TAGS["NTD_SatelliteNumber"], 's', 0, "\x00", True))

        if write_rgb:
            extra_tags.append((NINJO_TAGS["NTD_ColorDepth"], 'i', 1, 24, True))
        elif cmap:
            extra_tags.append((NINJO_TAGS["NTD_ColorDepth"], 'i', 1, 16, True))
        else:
            extra_tags.append((NINJO_TAGS["NTD_ColorDepth"], 'i', 1, 8, True))

        extra_tags.append(
            (NINJO_TAGS["NTD_DataSource"], 's', 0, data_source, True))
        extra_tags.append((NINJO_TAGS["NTD_XMinimum"], 'i', 1, 1, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_XMaximum"], 'i', 1, image_data.shape[1], True))
        extra_tags.append((NINJO_TAGS["NTD_YMinimum"], 'i', 1, 1, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_YMaximum"], 'i', 1, image_data.shape[0], True))
        extra_tags.append(
            (NINJO_TAGS["NTD_Projection"], 's', 0, projection, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_MeridianWest"], 'f', 1, meridian_west, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_MeridianEast"], 'f', 1, meridian_east, True))

        if radius_a is not None:
            extra_tags.append((NINJO_TAGS["NTD_EarthRadiusLarge"],
                               'f', 1, float(radius_a), True))
        if radius_b is not None:
            extra_tags.append((NINJO_TAGS["NTD_EarthRadiusSmall"],
                               'f', 1, float(radius_b), True))
        # extra_tags.append((NINJO_TAGS["NTD_GeodeticDate"], 's', 0, "\x00",
        # True)) # ---?
        if ref_lat1 is not None:
            extra_tags.append(
                (NINJO_TAGS["NTD_ReferenceLatitude1"], 'f', 1, ref_lat1, True))
        if ref_lat2 is not None:
            extra_tags.append(
                (NINJO_TAGS["NTD_ReferenceLatitude2"], 'f', 1, ref_lat2, True))
        if central_meridian is not None:
            extra_tags.append((NINJO_TAGS["NTD_CentralMeridian"],
                               'f', 1, central_meridian, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_PhysicValue"], 's', 0, physic_value, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_PhysicUnit"], 's', 0, physic_unit, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_MinGrayValue"], 'i', 1, min_gray_val, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_MaxGrayValue"], 'i', 1, max_gray_val, True))
        extra_tags.append((NINJO_TAGS["NTD_Gradient"], 'f', 1, gradient, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_AxisIntercept"], 'f', 1, axis_intercept, True))
        if altitude is not None:
            extra_tags.append(
                (NINJO_TAGS["NTD_Altitude"], 'f', 1, altitude, True))
        extra_tags.append((NINJO_TAGS["NTD_IsBlackLineCorrection"],
                           'i', 1, is_blac_corrected, True))
        extra_tags.append((NINJO_TAGS["NTD_IsAtmosphereCorrected"],
                           'i', 1, is_atmo_corrected, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_IsCalibrated"], 'i', 1, is_calibrated, True))
        extra_tags.append(
            (NINJO_TAGS["NTD_IsNormalized"], 'i', 1, is_normalized, True))
        extra_tags.append((NINJO_TAGS["NTD_TransparentPixel"],
                           'i', 1, transparent_pix, True))

        return args

    args = _create_args(image_data, pixel_xres, pixel_yres)

    tifargs = {}
    header_only_keys = ('byteorder', 'bigtiff', 'software', 'writeshape')
    for key in header_only_keys:
        if key in args:
            tifargs[key] = args[key]
            del args[key]

    if 'writeshape' not in args:
        args['writeshape'] = True
    if 'bigtiff' not in tifargs and \
            image_data.size * image_data.dtype.itemsize > 2000 * 2 ** 20:
        tifargs['bigtiff'] = True
    with local_tifffile.TiffWriter(output_fn, **tifargs) as tif:
        tif.save(image_data, **args)
        for _, scale in enumerate((2, 4, 8, 16)):
            shape = (image_data.shape[0] / scale,
                     image_data.shape[1] / scale)
            if shape[0] > tile_width and shape[1] > tile_length:
                args = _create_args(image_data[::scale, ::scale],
                                    pixel_xres * scale, pixel_yres * scale)
                for key in header_only_keys:
                    if key in args:
                        del args[key]
                tif.save(image_data[::scale, ::scale], **args)

    log.info("Successfully created a NinJo tiff file: '%s'" % (output_fn,))


# -----------------------------------------------------------------------------
#
# Read tags.
#
# -----------------------------------------------------------------------------
def read_tags(filename):
    """Will read tag, value pairs from Ninjo tiff file.

    :Parameters:
        filename : string
            Ninjo tiff file.

    :Return:
        A list tags, one tag dictionary per page.
    """
    pages = []
    with local_tifffile.TiffFile(filename) as tif:
        for page in tif:
            tags = {}
            for tag in page.tags.values():
                name, value = tag.name, tag.value
                try:
                    # Is it one of ours ?
                    name = int(name)
                    name = NINJO_TAGS_INV[name]
                except ValueError:
                    pass
                except KeyError:
                    name = tag.name
                tags[name] = value
            pages.append(tags)
    return pages


if __name__ == '__main__':
    import sys
    import getopt

    page_no = None
    print_color_maps = False
    opts, args = getopt.getopt(sys.argv[1:], "p:c")
    for key, val in opts:
        if key == "-p":
            page_no = int(val)
        if key == "-c":
            print_color_maps = True
    try:
        filename = args[0]
    except IndexError:
        print("""usage: python ninjotiff.py [<-p page-number>] [-c] <ninjotiff-filename>
    -p <page-number>: print page number (default are all pages).
    -c: print color maps (default is not to print color maps).""", sys.stderr)
        sys.exit(2)

    pages = read_tags(filename)
    if page_no is not None:
        try:
            pages = [pages[page_no]]
        except IndexError:
            print("Invalid page number '%d'" % page_no, sys.stderr)
            sys.exit(2)
    for page in pages:
        names = sorted(page.keys())
        print("")
        for name in names:
            if not print_color_maps and name == "color_map":
                continue
            print(name, page[name])
