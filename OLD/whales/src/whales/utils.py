from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.text import Text
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics.pairwise import euclidean_distances

from .chem_tools import GetCharge


def StackFingerprints(fps: Sequence[np.ndarray]):
    return np.stack(fps)


def ScaleFingerprints(df: pd.DataFrame, avg=None, std=None):
    avg = avg if avg is not None else df.mean()
    std = std if std is not None else df.std(ddof=-1)

    return (df - avg) / std, avg, std


def GetLabels(n: int, start=0, end=100, step=10, lab_string: str = ""):
    # produces labels strings
    strings = ["R_", "I_", "IR_"]
    perc = range(start, end + 1, step)

    labels = [f"{j}{lab_string}{int(i / 10)}" for j in strings for i in perc]

    return labels


def GetDataFrame(fps_matrix: np.ndarray):
    labels = GetLabels(fps_matrix.shape[1])
    return pd.DataFrame(fps_matrix, columns=labels)


def PlotDataFrame(df: pd.DataFrame):
    sns.set(rc={"figure.figsize": (16, 8.27)})  # sets the size of the boxplot
    sns.boxplot(data=df, linewidth=2)


def CalcSimilarityMatrix(a: np.ndarray, b: np.ndarray):
    return euclidean_distances(a, b)


def GetTopK(dist_matrix: np.ndarray, k: int = 10):
    # gets the top k most similar molecules
    sort_indices = np.argsort(dist_matrix)
    top_k = sort_indices[:, 0:k]
    return top_k


def GetMurckoScaffold(mol: Chem.Mol):
    # gets the Murcko scaffold
    return MurckoScaffold.GetScaffoldForMol(mol)


def GetMurckoScaffoldSmiles(mol: Chem.Mol):
    # gets the Murcko scaffold
    return MurckoScaffold.MurckoScaffoldSmiles(mol)


def GetFrequentScaffolds(suppl, output_type="supplier"):
    """Starting from a supplier file, the function computes the most frequently recurring scaffolds and returns them as a
    supplier file (if output_type='supplier') or as a counter file.
    """
    scaff_list = []
    for mol in suppl:
        scaff_list.append(GetMurckoScaffoldSmiles(mol=mol))

    freq_scaffolds = Counter()
    for scaff in scaff_list:
        freq_scaffolds[scaff] += 1

    freq_scaffolds = freq_scaffolds.most_common()

    if output_type == "supplier":
        # converts it back in a supplier file,
        suppl_new = []
        for row in freq_scaffolds:
            mol = Chem.MolFromSmiles(row[0])
            mol.SetProp(
                "_Name", str(round((row[1] / len(suppl)) * 100, 2)) + "%"
            )  # assigns the molecule name as the percentage occurrence
            suppl_new.append(mol)

        freq_scaffolds = suppl_new

    return freq_scaffolds


def PlotChargeMap(mol: Chem.Mol, fig_name=None, lab_atom=False, text=False, MapMin=0, MapMax=1):
    # settings
    scale = -1  # size of dots
    coordscale = 1  # coordinate scaling
    colmap = "bwr"

    mol, charge, err = GetCharge(mol, property_name="_GasteigerCharge", do_charge=True)

    if err:
        print("Error in charge calculation")

    n_atoms = mol.GetNumAtoms()  # num atoms
    charge = np.zeros((n_atoms, 1))  # init weights
    # coordinates and property
    for atom_ix in range(n_atoms):
        atom: Chem.Atom = mol.GetAtomWithIdx(atom_ix)
        value = atom.GetProp("_GasteigerCharge")
        charge[atom_ix] = value

    opts = Draw.DrawingOptions()
    opts.clearBackground = True
    opts.bgColor = (1, 1, 1)

    fig: Figure = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, charge, coordScale=coordscale, colorMap=colmap, colors="w", alpha=0, scale=scale
    )

    SimilarityMaps.Draw.MolDrawOptions.clearBackground

    if not lab_atom:
        for elem in fig.axes[0].get_children():
            if isinstance(elem, Text):
                elem.set_visible(False)

    plt.axis("off")

    if text:
        for atom_ix in range(n_atoms):
            x = mol._atomPs[atom_ix][0]
            y = mol._atomPs[atom_ix][1]
            plt.text(
                x,
                y,
                "%.2f" % charge[atom_ix],
                path_effects=[PathEffects.withStroke(linewidth=1, foreground="blue")],
            )

    if fig_name:
        fig.savefig(fig_name, bbox_inches="tight")

    return plt.show()
