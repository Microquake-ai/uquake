import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import offset_copy
from uquake.core import event

from uquake.core.logging import logger


def guttenberg_richter(magnitudes, dates, bin_size=0.05, b_range=[-2.0, -0.5],
                       magnitude_type='Moment magnitude', xlim=[-2.0, 1.0],
                       **kwargs):
    """
    produce a Guttenberg Richter plot from a list of magnitudes

    :param magnitudes: List of magnitudes
    :type magnitudes: list of float
    :param dates: list of event time associated with the magnitude
    :type dates: list of uquake.core.UTCDateTime
    :param bin_size: the width of bins
    :type bin_size: float
    :param b_range: range over which the b value is calculated an fit
    ([min, max])
    :type b_range: list containing two floats
    :param magnitude_type: Type of magnitude
    :type magnitude_type: str
    :param xlim: limits of the x axis
    :type xlim: list containing two floats
    """

    mag = np.array(magnitudes)

    wdt = 15
    num_years = (np.max(dates) - np.min(dates)) / (24 * 3600 * 365.25)

    bins = np.arange(xlim[0], xlim[1], bin_size)
    hist = np.histogram(mag, bins=bins)
    hist = hist[0][::-1]
    bins = bins[::-1]
    bins = bins[1::]
    cum_hist = hist.cumsum()
    cum_hist = np.array([float(nb) for nb in cum_hist])
    cum_hist /= float(num_years)

    new_cum_yearly_rate = []
    for i in cum_hist:
        new_cum_yearly_rate.append(i)

    log_cum_sum = np.log10(new_cum_yearly_rate)

    cum = np.zeros(bins.shape)
    min_mag = b_range[0]
    max_mag = b_range[1]
    indices = np.nonzero((bins >= min_mag) & (bins <= max_mag))[0]
    b, a = np.polyfit(bins[indices], log_cum_sum[indices], 1)

    mg = np.arange(min_mag, max_mag, 0.1)
    fitmg = b * mg + a

    # Tracer()()
    plt.semilogy(bins, new_cum_yearly_rate, 'k.', **kwargs)
    plt.semilogy(mg, 10 ** fitmg, 'k--',
                 label='best fit ($%0.1f\,M_w + %0.1f$)' % (b, a), **kwargs)
    plt.xlabel('%s' % magnitude_type)
    plt.ylabel('Number of events/year')
    plt.xlim(xlim)
    plt.legend()


# ylim = plt.ylim()
# plt.ylim([0, 10**0])

# plt.show()


class Plot(object):

    def __init__(self, data=None, picks=None, site=None):

        self.data = data
        self.picks = picks
        self.extraData = None
        self.style = "all"
        self.onlyIfHasPicks = True
        self.maxNumPlots = 999

        self.dataDirty = True
        self._plotData = None
        self.numPlots = 0
        self.numTraces = 0

        self.extraCaption = ""

    def close(self):

        # need to check that a plot has been created
        plt.close()

    def setData(self, data):

        for curTr, tr in enumerate(data):
            if curTr == 0:
                xmin = tr.stats.starttime
                xmax = tr.stats.endtime
            else:
                if tr.stats.starttime < xmin:
                    xmin = tr.stats.starttime
                if tr.stats.starttime > xmax:
                    xmax = tr.stats.endtime

        data.trim(starttime=xmin, endtime=xmax, pad=True, fill_value=0)

        self.data = data
        self.dataDirty = True

    def setPicks(self, picks):

        self.picks = picks
        self.dataDirty = True

    def _prepare_data(self):

        if not isinstance(self.picks, list):
            self.picks = [self.picks]

        self.data.detrend('demean')
        stations = np.unique([tr.stats.station for tr in self.data])

        self._plotData = []
        self.numTraces = 0
        for station in stations:
            curPicks = []
            # loop on picking stages
            for cat2 in self.picks:
                curStagePicks = []

                if isinstance(cat2, event.Event):
                    evts = cat2
                elif isinstance(cat2, event.Catalog):
                    if not cat2.events:
                        continue

                    evts = cat2.events
                else:
                    continue

                if not isinstance(evts, list):
                    evts = [evts]

                for evt in evts:
                    if not evt['picks']:
                        continue

                    prevPicks = evt['picks']

                    # find the existing P and S picks for the current station
                    for pick in prevPicks:
                        if pick['waveform_id'].station_code != station:
                            continue

                        curStagePicks.append(pick)

                    curPicks.append(curStagePicks)

            if self.onlyIfHasPicks:
                numPicks = np.sum([len(a) for a in curPicks])
                if numPicks == 0:
                    continue

            trs = self.data.select(station=station)
            # if len(trs) == 3:
            # trs = trs.rotate_P_S()

            self._plotData.append({'traces': trs, 'picks': curPicks})
            self.numTraces += len(trs)

        self.numPlots = 0
        if self.style == "all":
            self.numPlots = self.numTraces
            if self.numPlots > self.maxNumPlots:
                self.numPlots = self.maxNumPlots

        self.dataDirty = False

    def _make_plot(self, ax, plt_data, max_val, curPicks, starttime,
                   cur_starttime, sr, transOffset, caption=None):

        npts = len(plt_data)

        t = np.array(
            [(starttime + tmp).datetime for tmp in np.arange(0, npts) / sr])
        ax.plot(t, plt_data / max_val)
        plt.ylim([-1, 1])
        if caption:
            ax.text(.95, .9, caption, transform=ax.transAxes, va='top',
                    ha='right', fontsize=10, backgroundcolor='white')

        for curStage, pickStage in enumerate(curPicks):
            for pick in pickStage:
                pick_sample = pick.time.datetime
                col = 'red' if pick['phase_hint'] == 'P' else 'black'
                # col = 'red'
                ax.axvline(pick_sample, c=col)

                snr = None
                for c in pick.comments:
                    if 'SNR=' not in c.text:
                        continue

                    snr = c.text.split('=')[1]

                displayText = '%s%s' % (pick.phase_hint, curStage)
                if snr:
                    displayText = '%s - %s' % (displayText, snr)

                label = Text(pick_sample,
                             ax.get_ylim()[1] * .7, displayText,
                             color=col, backgroundcolor='white', size=10,
                             alpha=.8, transform=transOffset)
                ax.add_artist(label)

                if hasattr(pick, 'tt_residual'):
                    pick_sample = (pick.time - pick.tt_residual).datetime
                    ax.axvline(pick_sample, c='green')

    # This function supports plotting several stages of picking on the same graph.
    # Simply pass in an array of catalogs to see the pick progression from one stage to the other.
    def plot(self):

        if self.dataDirty:
            self._prepare_data()
        if (self.style == "all" and self.numPlots == 0) \
                or not self._plotData:
            logger.warning('No data to plot!')
            return

        fig = None

        if self.style == "all":
            if self.extraData:
                self.numPlots += 1

            fig = plt.figure(figsize=(10, 2 * self.numPlots), dpi=100)

        plotOffset = 0

        for curSt, t in enumerate(self._plotData):
            trs = t['traces']
            curPicks = t['picks']

            if self.style == "per_station":
                self.numPlots = len(trs)
                if self.extraData:
                    self.numPlots += 1
                fig = plt.figure(figsize=(10, 5 * self.numPlots), dpi=100)

            starttime = trs[0].stats.starttime
            sr = trs[0].stats.sampling_rate
            cur_starttime = starttime
            transOffset = None

            curTr = 0

            for curTr, tr in enumerate(trs):
                curPlot = plotOffset + curTr
                if curPlot >= self.maxNumPlots:
                    # finished
                    return

                ax = plt.subplot(self.numPlots, 1, curPlot + 1)
                caption = '%s - %s' % (tr.stats.station, tr.stats.channel)
                if self.extraCaption:
                    caption = '%s - %s' % (caption, self.extraCaption[curSt])

                transOffset = offset_copy(ax.transData, fig=fig,
                                          x=5, y=0, units='points')
                max_val = np.max(np.abs(tr.data))
                if max_val == 0:
                    continue

                cur_starttime = tr.stats.starttime
                self._make_plot(ax, tr.data, max_val, curPicks, starttime,
                                cur_starttime, sr, transOffset,
                                caption=caption)
                plt.title('station %s' % tr.stats.station)

            if self.style == "per_station":
                if self.extraData:
                    ax = plt.subplot(self.numPlots, 1, self.numPlots)
                    max_val = np.max(np.abs(self.extraData[curSt]))

                    self._make_plot(ax, self.extraData[curSt], max_val,
                                    curPicks, starttime, cur_starttime, sr,
                                    transOffset)

                self.show()
            else:
                plotOffset += curTr + 1

    def show(self):

        plt.show()

    def saveFig(self, outFile=""):

        plt.tight_layout()
        plt.savefig(outFile, dpi=100, bbox_inches='tight')
