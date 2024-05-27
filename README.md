# WUCSS

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Last Commit](https://img.shields.io/github/last-commit/Roche/WUCSS/main?style=flat-square)](https://github.com/Roche/WUCSS/commits/main)
[![Python Version](https://img.shields.io/badge/python-3.8-blue)](README.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Workflow for Unsupervised Clustering of Sleep States (WUCSS) is a novel workflow designed for unsupervised clustering of sleep states in rodents. It leverages accelerometer and electrophysiological data to classify different sleep states with high accuracy. The method enhances traditional sleep staging by introducing an unbiased approach, particularly excelling in discriminating between deep and light sleep states.

## Table of Contents

- [About](#about)
- [Motivation and Documentation](#documentation)
- [Dependencies](#dependencies)
- [Dataset requirements](#dataset_req)
- [Getting Started](#getting_started)
- [License](#license)
- [Contributing](#contributing)
- [Contributors](#contributors)

## <a id="about"></a> About

WUCSS is part of a larger Python workflow for analysing all kinds of electrophysiological data. The code presented here is tailored to run on the example dataset. All variables and directories are set up to run without any modification. This is to provide an example of how the code can be applied and to show the results the module produces. To use this module for other datasets, you will need to adjust some of the parameters. 

## <a id="documentation"></a> Motivation and Documentation

The module is published in the [Journal of Neuroscience Methods](https://www.sciencedirect.com/science/article/pii/S0165027024001006). 

The repository complements the manuscript and provides a better understanding of how the WUCSS module works. It is a functional standalone module that can be used and modified by others who are interested. The dataset we provide is used in the manuscript to generate Figures 2 and 3. We hope that by providing the code for the module, other groups will be inspired, explore the module and ideally gain a better understanding of their data.

## <a id="dependencies"></a> Dependencies

The WUCSS module is written in Python, and requires Python >= 3.8 to run.

It has the following required dependencies:

- os
- json
- math
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [nex](https://pypi.org/project/nex/)
- [scipy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [functools](https://pypi.org/project/functools/)
- [seaborn](https://pypi.org/project/seaborn/)

We further provide a .yml file to quickly set up the Python environment for the module ([.yml file](wucss_env.yml)). The file contains all the Python libraries required for the module.

We recommend using the [Anaconda](https://www.anaconda.com/download-2) distribution to manage these requirements.

## <a id="dataset_req"></a> Dataset requirements
- File format:
We provide functions that allow Python to handle the .nex5 file format, if other file formats are used, the code must be adapted accordingly. The .nex5 file format allows continuous data, such as an EEG signal, to be stored with annotated intervals, such as time stamps of when the animal was awake or asleep. 

- Electrode placement:
Sleep classification was tested on EEG data from frontal and parietal EEG. Signals coming from other brain areas were not tested to classify the sleep state. 

- Behavioural state annotations:
The module starts at a point where it expects the data to be annotated into awake and sleep states, therefore the file or another data source must provide timestamps of such annotations.

## <a id="getting_started"></a> Getting Started

This repository runs on the example dataset. The steps to get the module to run on this example dataset are: 

1) Download the repository
2) Install the dependencies, e.g. in an anaconda environment. 
3) Run the [main task](wucss_main_task.py) of WUCSS
4) View results 
    * [WUCSS Cluster](results/Cntnap001_200504_wucss_qc.png)
    * [WUCSS Hypnogram](results/Hypnogram.png) 
    * [WUCSS Epochs](results/start_and_ends_intervals.csv)
      
**Note**: the example file is quite heavy (around 500 MB), as it represents a 24-hour recording. So both downloading and running the example will take some time, around 10-20 minutes depending on your PC.

## <a id="license"></a> License 

WUCSS uses the Apache License 2.0. 

The Apache License, Version 2.0, is a permissive open-source license that allows for the free use, modification, and distribution of software. It also includes patent grants, protecting users from patent litigation related to the software. The license is well-known and widely used in the open-source community.

## <a id="contributing"></a> Contributing

First off, thanks for taking the time to contribute! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are greatly appreciated.

Please read our [contribution guidelines](CONTRIBUTING.md), and thank you for being involved!

## <a id="contributors"></a> Contributors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/grosss10-roche">
            <img src="https://avatars.githubusercontent.com/u/151529315?v=4" width="100;" alt="Roche"/>
            <br />
            <sub><b>Simon Gross</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/cusinatr">
            <img src="https://avatars.githubusercontent.com/u/74003718?v=4" width="100;" alt="Roche"/>
            <br />
            <sub><b>Riccardo Cusinato</b></sub>
        </a>
    </td></tr>
</table>
