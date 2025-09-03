# ======================================================================================================================
# * Weighted Holistic Atom Localization and Entity Shape (WHALES) descriptors *
#   v. 1, May 2018
# ----------------------------------------------------------------------------------------------------------------------
# This file contains all the necessary functions to calculate WHALES descriptors for the
# molecules contained in an rdkit supplier.
#
# Francesca Grisoni, May 2018, ETH Zurich & University of Milano-Bicocca, francesca.grisoni@unimib.it
# please cite as:
#   Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A. Hiss, Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider
#   "Scaffold hopping from natural products to synthetic mimetics by holistic molecular similarity",
#   Nature Communications Chemistry 1, 44, 2018.
# ======================================================================================================================
from __future__ import annotations

import numpy as np
from rdkit import Chem

from .chem_tools import GetCoordinatesAndProps
from .lcm import lmahal


# ----------------------------------------------------------------------------------------------------------------------
def GetFingerprint(mol: Chem.Mol, charge_threshold=0, do_charge=True, property_name=""):
    # check for correct molecule import, throw an error if import/sanitization fail

    mol, err = ImportMol(mol)

    if err:
        x = np.full((33,), -999.0)
        print("Molecule not loaded.")

    else:
        # coordinates and partial charges (checks for computed charges)
        coords, w, err = GetCoordinatesAndProps(mol, property_name, do_charge)
        if not err:  # no errors in charge
            # does descriptors
            x = DoLCD(coords, w, charge_threshold)
        else:
            x = np.full((33,), -999.0)
            print("No computed charges.")

    return x


def ImportMol(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    # initialization
    err = False

    if mol is None:
        err = True

    else:
        # sanitize
        sanit_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanit_fail:
            raise ValueError(sanit_fail)
            err = True

    return mol, err


# ----------------------------------------------------------------------------------------------------------------------
def DoLCD(coords, w, thr: float):
    """Core function for computing 3D LCD descriptors, starting from the coordinates and the partial charges.
    :param coords: molecular 3D coordinate matrix (n_at x 3)
    w(n_at x 1): molecular property to consider
    :param w: partial charges
    :param lcm_thr: threshold to be used to retain atoms (e.g., 0.001)
    :return:
    x_all: descriptors  for the molecules (1 x p)
    """
    # calculates lcm with weight scheme 1 (all charges)
    res = lmahal(coords, w)

    # applies sign
    res = ApplySign(w, res, thr)

    x_all = ExtractLCM(res)  # MDs

    return x_all


# ----------------------------------------------------------------------------------------------------------------------
def ApplySign(w, res, thr=0.0):
    """Applies the sign to negatively charged atoms.
    :param w: partial charge
    :param res: computed atomic descriptors
    :param thr: threshold to consider atoms as negatively charged (default is 0); other atoms are removed
    :return: computed atomic descriptors with adjusted sign
    """
    # find negative weights and assigns a "-"
    a, _ = np.where(w < 0)
    res[a, :] *= -1

    # removes atoms with abs(w) smaller than the thr
    a, _ = np.where(abs(w) < thr)
    res = np.delete(res, a, 0)

    return res


# ----------------------------------------------------------------------------------------------------------------------
def ExtractLCM(data, start=0, end=100, step=10):
    """Extracts descriptors referred to the whole molecule from numbers referred to atoms, e.g., R and I.
    ====================================================================================================================
    :param:
    data (n_atom x p): atomic description
    start (int): minimum percentile (default = minimum value)
    end (int): maximum percentile (default = maximum value)
    step (int): step for percentiles generation (default, 10 corresponds to deciles)
    :returns
    x(1 x p1): molecular description based on percentiles
    labels(1 x p1): descriptor labels
    ====================================================================================================================
    """
    # Calculates percentiles according to the specified settings
    perc = range(start, end + 1, step)
    data = np.array(data)
    x = np.percentile(data, list(perc), axis=0)
    x = np.concatenate((x[:, 0], x[:, 1], x[:, 2]), axis=0)  # Flattens preserving the ordering

    # rounds the descriptors to the third decimal place
    x = np.round(x, 3)

    return x
