"""
Functions and class to read data in nex5 file

written by Simon Gross Nov-2023
"""

import json
import os.path

import numpy as np
from nex.nexfile import NexFileVarType, Reader


class Nex5CustomReader(Reader):
    """
    class to read nex5 files in a custom way, in order to read some or all variables:
    """

    def __init__(self, useNumpy=True):
        super(Nex5CustomReader, self).__init__(useNumpy=useNumpy)

    def custom_ReadNex5File(self, filePath, variables = None):
        """
        funtion to reads data from .nex5 file. If variables is None it reads all variables. If it is a list of variable names, it reads only those variables.

        :param filePath: str: full path of file
        :param variables: list: with variable names to read

        :return: file data dict with information of nex5 file
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == '.nex':
            raise NotImplementedError("only nex5 files are implemented at the moment")

        self.fileData = {}
        self.theFile = open(filePath, 'rb')

        # read file header
        self.fileData['FileHeader'] = self._ReadNex5FileHeader()
        #here a difference with the original, we use a dict instead of list for Variables
        self.fileData['Variables'] = dict()
        # Another difference, we store the data type of each variable
        VariablesMetaData = dict()

        # read variable headers and create variables
        for varNum in range(self.fileData['FileHeader']['NumVars']):
            var = {}
            var['Header'] = self._ReadNex5VarHeader()
            var_name = var['Header']['Name']
            #this is also different from the original
            #we append the variable if the name has been requested or everything has been requested (variables is None)
            if variables is None or var_name in variables:
                self.fileData['Variables'][var_name] = var

            # We also keep the variable data type as string
            varType = var['Header']['Type']
            if varType == NexFileVarType.NEURON:
                varType_str = "neuron"
            elif varType == NexFileVarType.EVENT:
                varType_str = "event"
            elif varType == NexFileVarType.INTERVAL:
                varType_str = "interval"
            elif varType == NexFileVarType.WAVEFORM:
                varType_str = "wave"
            elif varType == NexFileVarType.POPULATION_VECTOR:
                varType_str = "pop_vector"
            elif varType == NexFileVarType.CONTINUOUS:
                varType_str = "continuous"
            elif varType == NexFileVarType.MARKER:
                varType_str = "marker"
            VariablesMetaData[var_name] = varType_str

        # read variable data
        self._ReadData()

        # read metadata
        metaOffset = self.fileData['FileHeader']['MetaOffset']
        if metaOffset > 0:
            self.theFile.seek(0, os.SEEK_END)
            size = self.theFile.tell()
            if metaOffset < size:
                self.theFile.seek(metaOffset)
                metaString = self.theFile.read(size - metaOffset).decode('utf-8').strip('\x00')
                metaString = metaString.strip()
                try:
                    self.fileData['MetaData'] = json.loads(metaString)
                    # If metadata exist, add also data type
                    for varNum in range(len(self.fileData['MetaData']['variables'])):
                        var_name = self.fileData['MetaData']['variables'][varNum]['name']
                        varType = VariablesMetaData[var_name]
                        self.fileData['MetaData']['variables'][varNum]['type'] = varType

                except Exception as error:
                    print(('Invalid file metadata: ' + repr(error)))

        self.theFile.close()
        return self.fileData

    def _ReadData(self):
        for var_name, var in iter(self.fileData['Variables'].items()):
            self.theFile.seek(var['Header']['DataOffset'])
            varType = var['Header']['Type']
            if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
                self._ReadTimestamps(var)
            elif varType == NexFileVarType.INTERVAL:
                self._ReadIntervals(var)
            elif varType == NexFileVarType.WAVEFORM:
                self._ReadWaveforms(var)
            elif varType == NexFileVarType.POPULATION_VECTOR:
                self._ReadPopVectors(var)
            elif varType == NexFileVarType.CONTINUOUS:
                self._ReadContinuous(var)
            elif varType == NexFileVarType.MARKER:
                self._ReadMarker(var)

def read_continuous_variables(fpath, channel_names):
    """
    Reads a list of continuous variables with names in the list channel_names.
    It returns the continuos data, the timestamps and the acquisition frequency
    of the first channel (assuming that all channels have the same acquisiton freq).
    It also returns the timestamps of the first point in each fragment, see fragment_tstamps for explanation.

    :param fpath: str: path to the nex5 file
    :param channel_names: list: with names of the variables to read

    :return: datamat array with N channels x samples,
        final_tstamps array with timestamps
        var_freq float with acquisition frequency of the first channel
        fragments_tstamps list with timestamp to the first data point in each fragment. The first element is the same
            as the first element in final_tstamps. If there is more than 1 element, it means there was an
            interruption in the data acquisition, there is a gap in the
            signal and each element points at the timestamp of the first datapoint after the gap(s).
            In this case final_tstamps will have a jump at some point.
    """

    # first read the countinous variables
    reader = Nex5CustomReader(useNumpy=True)
    data = reader.custom_ReadNex5File(fpath, channel_names)

    # from the first channel get the frequency and other information to build the timestamps
    var_data1 = data['Variables'][channel_names[0]]
    var_freq = var_data1["Header"]["SamplingRate"]
    fragments_counts = var_data1["FragmentCounts"]
    fragments_indexes = var_data1["FragmentIndexes"]
    fragments_tstamps = var_data1["FragmentTimestamps"]

    # build the timestamps
    for indx, fragindx in enumerate(fragments_indexes):
        offset = fragments_tstamps[indx]
        npoints = fragments_counts[indx]
        space = np.linspace(0, npoints - 1, npoints) * (1 / var_freq)
        tstamps = space + offset
        if indx == 0:
            final_tstamps = np.copy(tstamps)
        else:
            final_tstamps = np.append(final_tstamps, tstamps)

    # now get all the continous values and build the datamat
    npoints_all = var_data1["Header"]["NPointsWave"]
    datamat = np.zeros((len(channel_names), npoints_all))
    var_data = data['Variables']
    for indx, chan_name in enumerate(channel_names):
        var = var_data[chan_name]
        contvalues = np.asarray(var['ContinuousValues'])
        datamat[indx, :] = contvalues

    return datamat, final_tstamps, var_freq, fragments_tstamps


def read_interval_variables(fpath, interval_names):
    """
    function to read interval data (starts and ends)

    :param fpath: str: path to the nex5 file
    :param interval_names: list: of names of the intervals

    :return: interval_data dict with interval name as key and as value a tuple of (interval starts, interval ends)
    """

    # now read the interval variables
    reader = Nex5CustomReader(useNumpy=True)
    data = reader.custom_ReadNex5File(fpath, interval_names)

    # extract the start timestamps and the end timestamps
    interval_data = dict()
    var_data = data['Variables']
    for indx, interv_name in enumerate(interval_names):
        var = var_data[interv_name]
        interval_starts, interval_ends = var["Intervals"]
        interval_data[interv_name] = (interval_starts, interval_ends)
    return interval_data


def read_event_variables(fpath, event_names):
    """
    function to read event data (timestamps)

    :param fpath: str: with path to the nex5 file
    :param event_names: list: of names of the events

    :return: event_data dict with event name as key and timestamp as value
    """

    # now read the interval variables
    reader = Nex5CustomReader(useNumpy=True)
    data = reader.custom_ReadNex5File(fpath, event_names)

    # extract the start timestamps and the end timestamps
    event_data = dict()
    var_data = data['Variables']
    for indx, event_name in enumerate(event_names):
        var = var_data[event_name]
        curevents = var["Timestamps"]
        event_data[event_name] = curevents

    return event_data


def calculate_interval_durations(fpath, interval_names):
    """
    function to calculate the total duration of a group of intervals, the nex5 file is going to be read directly without NeuroExplorer

    :param fpath: str: of path to the nex5 file
    :param interval_names: list: of names of intervals

    :return: interval_durations dict with interval name as key and duration as value
    """

    #read the intervals
    interval_data = read_interval_variables(fpath, interval_names)
    interval_durations = dict()
    for interval_name, (starts, ends) in iter(interval_data.items()):
        interval_values = zip(starts, ends)
        total_duration = 0
        for start, end in interval_values:
            total_duration += end - start
        interval_durations[interval_name] = total_duration
    return interval_durations


def exists_variable(fpath, varname):
    """
    function to returns True if an variable exists, False otherwise

    :param: fpath: string: path to the nex5 file
    :param: varname: string: name of the intervals to check

    :return: result boolean with True if variable exists
    """

    # now read the interval variables
    reader = Nex5CustomReader(useNumpy=True)
    result = False
    try:
        data = reader.custom_ReadNex5File(fpath, [varname])
        var_data = data['Variables']
        if var_data:
            result = True
    except:
        pass

    return result


def exclude_interval_from_other_interval(doc, original_interval, interval_to_exclude, dic_itvs=False):
    """
    function to exclude time periods (interval_to_exclude) from another interval (original_interval)

    :param doc: str: path to file
    :param original_interval: str: name of interval from that we exclude the interval_to_exclude
    :param interval_to_exclude: str: name of interval that will be excluded
    :param dic_itvs: dict: with interval names as key and list of starts and ends as value

    :return: new_interval array with a list of starts and a list of ends
    """
    if not dic_itvs:
        dic_itvs = read_interval_variables(doc, [original_interval,interval_to_exclude])
    if interval_to_exclude:
        new_starts, new_ends = [], []
        for idx1 in range(dic_itvs[original_interval][0].shape[0]):
            start_org, end_org = dic_itvs[original_interval][0][idx1], dic_itvs[original_interval][1][idx1]
            new_starts.append(start_org)
            new_ends.append(end_org)
            for idx2 in range(dic_itvs[interval_to_exclude][0].shape[0]):
                start_exc, end_exc = dic_itvs[interval_to_exclude][0][idx2], dic_itvs[interval_to_exclude][1][idx2]
                if start_org <= start_exc and end_org >= start_exc:
                    if start_org == start_exc:
                        new_starts.remove(start_org)
                    else:
                        new_ends.append(start_exc)
                    if end_org > end_exc:
                        new_starts.append(end_exc)
                    else:
                        new_ends.remove(end_org)
                if end_org >= end_exc and start_org >= start_exc and start_org <= end_exc:
                    new_starts.remove(start_org)
                    new_starts.append(end_exc)
                if start_org >= start_exc and end_org <= end_exc:
                    new_starts.remove(start_org)
                    new_ends.remove(end_org)
        new_starts.sort()
        new_ends.sort()
        new_interval = [new_starts, new_ends]
    else:
        new_interval = [dic_itvs[original_interval][0], dic_itvs[original_interval][1]]

    return new_interval