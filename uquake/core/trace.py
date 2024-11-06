# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: Expansion of the obspy.core.trace module
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.trace module

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from abc import ABC
import numpy as np
from obspy.core.trace import Trace as ObspyTrace
from obspy.core.trace import Stats as ObspyStats
from obspy import UTCDateTime
from obspy.core.event import WaveformStreamID
from obspy.core.trace import AttribDict
from uquake.core.util.decorators import update_doc

from .event import Pick
from .util import tools
from scipy.ndimage import gaussian_filter
from uquake.core.logging import logger
from scipy.signal import detrend
from .util.signal import WhiteningMethod, GaussianWhiteningParams


@update_doc
class Stats(ObspyStats, ABC):
    __doc__ = ObspyStats.__doc__.replace('obspy', 'uquake')

    def __init__(self, stats=None, **kwargs):
        super().__init__(**kwargs)

        if stats:
            for item in stats.__dict__.keys():
                self.__dict__[item] = stats.__dict__[item]

        self.is_one_bit = False

    @property
    def instrument(self):
        return f'{self.station}.{self.location}'

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)


class Trace(ObspyTrace, ABC):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)

        if trace:
            self.stats = Stats(stats=trace.stats)
            self.data = trace.data

        elif 'header' in kwargs.keys():
            self.stats = Stats(stats=kwargs['header'])

        elif 'stats' in kwargs.keys():
            self.stats = Stats(stats=kwargs['stats'])

        else:
            self.stats = Stats()

    @property
    def sr(self):
        return self.stats.sampling_rate
    #
    # @property
    # def ppv(self):
    #     return np.max(np.abs(self.data))

    # @property
    # def ppa(self):
    #     return None

    @property
    def instrument(self):
        # assuming that all traces are from the same network
        return self.stats.instrument

    @property
    def instrument_code(self):
        return self.instrument

    def time_to_index(self, time):
        return int((time - self.stats.starttime) * self.sr)

    def index_to_time(self, index):
        return self.stats.starttime + (index / self.sr)

    def times(self):
        sr = self.stats.sampling_rate

        return np.linspace(0, len(self.data) / sr, len(self.data))

    def plot(self, **kwargs):
        from uquake.imaging.waveform import WaveformPlotting
        from uquake.core.stream import Stream
        waveform = WaveformPlotting(stream=self, **kwargs)

        return waveform.plotWaveform()

    def make_pick(self, pick_time, wlen_search,
                  stepsize, snr_wlens, phase_hint=None):

        ipick = self.time_to_index(pick_time)
        sr = self.stats.sampling_rate
        stepsize_samp = int(stepsize * sr)
        snr_wlens_samp = (snr_wlens * sr).astype(int)
        wlen_search_samp = int(wlen_search * sr)

        new_pick, snr = tools.repick_using_snr(self.data, ipick,
                                               wlen_search_samp, stepsize_samp,
                                               snr_wlens_samp)

        waveform_id = WaveformStreamID(channel_code=self.stats.channel,
                                       station_code=self.stats.station)

        pick = Pick(time=self.index_to_time(newpick), waveform_id=waveform_id,
                    phase_hint=phase_hint, evaluation_mode='automatic',
                    evaluation_status='preliminary', method='snr', snr=snr)

        return pick

    def time_within(self, utime, edge_buf=0.0):
        within = True

        if (utime - edge_buf) < self.stats.starttime:
            within = False
        elif (utime + edge_buf) > self.stats.endtime:
            within = False

        return within

    @classmethod
    def create_from_json(cls, trace_json_object):

        trace_json_object['stats']['starttime'] = UTCDateTime(
            trace_json_object['stats']['starttime'])
        trace_json_object['stats']['endtime'] = UTCDateTime(
            trace_json_object['stats']['endtime'])

        trc = cls(header=trace_json_object['stats'], data=np.array(
            trace_json_object['data'], dtype='float32'))

        return trc

    def to_json(self):
        trace_dict = dict()
        trace_dict['stats'] = dict()

        for key in self.stats.keys():
            if isinstance(self.stats[key], UTCDateTime):
                trace_dict['stats'][key] = self.stats[key].isoformat()
            elif isinstance(self.stats[key], AttribDict):
                trace_dict['stats'][key] = self.stats[key].__dict__
            else:
                trace_dict['stats'][key] = self.stats[key]

        trace_dict['data'] = self.data.tolist()

        return trace_dict

    def create_brune_pulse(self, f_c, shift_time):
        """
        Generates the Brune pulse for a given corner frequency and time array.

        Parameters:
        f_c (float): Corner frequency in Hz.
        shit_time (numpy array): shift the start of the trace to shift_time
        Returns:
        numpy array: The Brune pulse evaluated at the given time points.
        """
        times = self.times()
        # Calculate the decay constant
        tau = 1 / (2 * np.pi * f_c)

        # Generate the Brune pulse
        pulse = times * np.exp(-times / tau)

        pulse = np.roll(pulse, self.time_to_index(shift_time))

        pulse /= np.linalg.norm(pulse)

        return pulse

    def convert_to_one_bit(
            self, whitening_method: WhiteningMethod = WhiteningMethod.Gaussian,
            params: GaussianWhiteningParams = None
    ):
        """
        Convert the data to a one-bit representation by detrending, whitening,
        and then applying a sign function.

        This method prepares the data by removing any linear and constant trends,
        applying spectral whitening, and finally converting the signal to a one-bit
        representation. The one-bit conversion sets all positive values to 1 and all
        negative values to -1, resulting in a binary amplitude spectrum.

        Parameters:
        ----------
        whitening_method : WhiteningMethod, optional
            Specifies the whitening approach to use before one-bit conversion.
            Options include:
            - WhiteningMethod.Gaussian: Uses a Gaussian filter to smooth the amplitude
              spectrum.
            - WhiteningMethod.Basic: Sets all amplitudes to 1, preserving only phase
              information.
            Default is WhiteningMethod.Gaussian.

        params : GaussianWhiteningParams, optional
            Parameters for Gaussian whitening, used if `whitening_method` is set to
            Gaussian. If not provided, default values are used. Relevant parameters
            include:
            - smoothing_kernel_size: Standard deviation for the Gaussian filter.
            - water_level: Stabilization constant added to avoid division by zero.

        Returns:
        -------
        self : instance
            Returns the instance with `self.data` updated to contain the one-bit
            representation.

        Notes:
        -----
        - Detrending is applied twice: first to remove a linear trend and then a constant
          offset to ensure the data is centered around zero.
        - Whitening is applied based on the specified `whitening_method` to normalize
          the spectral components before one-bit conversion.
        - The final step, `np.sign(self.data)`, assigns 1 to positive values and -1 to
          negative values in the data, emphasizing signal zero-crossings and reducing
          amplitude variations.

        Example:
        --------
        ```python
        # Convert data to one-bit with default whitening
        instance.convert_to_onebit()

        # Convert data to one-bit with custom Gaussian whitening parameters
        custom_params = GaussianWhiteningParams(smoothing_kernel_size=1.5,
        water_level=0.02)
        instance.convert_to_onebit(whitening_method=WhiteningMethod.Gaussian,
        params=custom_params)
        ```
        """
        self.data = detrend(detrend(self.data, type='linear'), type='constant')
        self.whiten(method = whitening_method, params = params)
        self.data = np.sign(self.data)


    def whiten(self, method: WhiteningMethod = WhiteningMethod.Gaussian,
               params: GaussianWhiteningParams = GaussianWhiteningParams()):
        """
        Apply spectral whitening to the data, setting the amplitude of the frequency
        components to a uniform or smoothed distribution while preserving phase
        information.

        Parameters:
        ----------
        method : WhiteningMethod, optional
            Specifies the whitening approach. Options include:
            - WhiteningMethod.Gaussian: Uses a Gaussian filter to smooth the amplitude
              spectrum, dividing the data by the smoothed spectrum plus a water level.
            - WhiteningMethod.Basic: Sets all amplitudes to 1, preserving only phase
              information.
            Default is WhiteningMethod.Gaussian.

        params : GaussianWhiteningParams, optional
            Parameters for the Gaussian smoothing method. If not provided, default values
            from GaussianWhiteningParams are used. Relevant parameters include:
            - smoothing_kernel_size: Standard deviation for the Gaussian filter to
              control smoothing of the amplitude spectrum.
            - water_level: A small constant added to avoid division by zero and stabilize
              the spectrum.

        Returns:
        -------
        self : instance
            Returns the instance with `self.data` updated to contain the whitened
            time-domain data.

        Notes:
        -----
        - When using the Gaussian method, the amplitude spectrum of `data_fft` is first
          smoothed with a Gaussian filter. The original data is then normalized by the
          smoothed spectrum plus a water level for stability.
        - When using the Basic method, the amplitude of each frequency component is set
          to 1, preserving the original phase.
        - If `params` is None and Gaussian whitening is selected, default values are
          logged and used for `smoothing_kernel_size` and `water_level`.
        - The operation is performed in place

        Example:
        --------
        ```python
        # Apply Gaussian whitening with custom parameters
        custom_params = GaussianWhiteningParams(smoothing_kernel_size=2.0, water_level=0.05)
        whitened_instance = instance.whiten(method=WhiteningMethod.Gaussian, params=custom_params)
        ```
        """

        data = self.data
        data_fft = np.fft.fft(data)
        if method == WhiteningMethod.Gaussian:
            if params is None:
                logger.warning(
                    f'parameter were not provided... '
                    f'proceeding with default values\n'
                    f'smoothing kernel: '
                    f'{GaussianWhiteningParams().smoothing_kernel_size}\n'
                    f'water level: {GaussianWhiteningParams().water_level}')
            smooth_spectrum = gaussian_filter(
                np.abs(data_fft), sigma=params.smoothing_kernel_size)
            data_fft /= (smooth_spectrum + params.water_level)
        else:
            data_fft = np.exp(-1j * np.angle(data_fft))

        self.data = np.real(np.fft.ifft(data_fft))



