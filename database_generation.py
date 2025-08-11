"""
Generate and save database containing basic information, stats, and indices for each cell.
"""

import os
from scipy import stats
from jaratoolbox import celldatabase
from jaratoolbox import histologyanalysis as ha
from jaratoolbox import spikesorting
from jaratoolbox import settings
import database_generation_funcs as funcs
import params
import sys


def calculate_base_stats(db):
    return db


def calculate_indices(db):
    return db


def calculate_cell_locations(db):
    brainAreaDict = None  # {'left_AudStr': 'LeftAstr', 'right_AudStr': 'RightAstr'}
    filterConditions = None
    celldb = ha.cell_locations(db, filterConditions, brainAreaDict)
    return celldb


if __name__ == "__main__":

    # -- Spike sort the data (code is left here for reference) --
    '''
    subject = 'testXXX'
    inforecFile = os.path.join(settings.INFOREC_PATH,'{}_inforec.py'.format(subject))
    clusteringObj = spikesorting.ClusterInforec(inforecFile)
    clusteringObj.process_all_experiments()
    '''

    # Generate cell database (this function excludes clusters with isi>0.05, spikeQuality<2
    celldb = celldatabase.generate_cell_database_from_subjects(params.subject_list)

    # Calculate base stats and indices for each cell
    celldb = calculate_base_stats(celldb)  # Calculated for all cells
    celldb = calculate_indices(celldb)  # Calculated for a selected subset of cells
    celldb = calculate_cell_locations(celldb)  # TODO: Need jarashare/histology for this

    dbPath = os.path.join(settings.FIGURES_DATA_PATH, params.STUDY_NAME) # TODO: Need jarahubdata/figuresdata for this
    dbPath2022 = os.path.join(settings.FIGURES_DATA_PATH, params.STUDY_NAME_2022)
    dbFilename = os.path.join(dbPath, 'celldb_{}.h5'.format(params.STUDY_NAME))
    dbFilename2022 = os.path.join(dbPath2022, 'celldb_{}.h5'.format(params.STUDY_NAME_2022))

    if os.path.isdir(dbPath):
        celldatabase.save_hdf(celldb, dbFilename)
        print('Saved database to {}'.format(dbFilename))
    else:
        print('{} does not exist. Please create this folder.'.format(dbPath))

    if os.path.isdir(dbPath2022):
        celldatabase.save_hdf(celldb, dbFilename2022)
        print('Saved database to {}'.format(dbFilename2022))
    else:
        print('{} does not exist. Please create this folder.'.format(dbPath2022))
