# %%
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def PrepareMol(
    mol: Chem.Mol,
    do_geometry: bool = True,
    do_charge: bool = True,
    property_name: str = "_GasteigerCharge",
    max_iter: int = 1000,
    mmffvariant: str = "MMFF94",
    seed: int = 26,
    max_attempts: int = 5,
) -> Chem.Mol:
    """Sets atomic properties if they are specified in the sdf, otherwise computes them. If specified, computes 3D coordinates
    using MMF.  The default number of iterations is 200, but it is progressively increased to 5000 (with a step of 500)
    in case convergence is not reached.
    ====================================================================================================================
    :param
    mol: molecule to be analyzed (from rdkit supplier)
    property_name: name of the property to be used
    do_charge: if True, partial charge is computed
    do_geom: if True, molecular geometry is optimized
    :return:
    mol: molecule with property and 3D coordinates (H depleted)
    property_name: updated on the basis of the settings
    ====================================================================================================================
    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich

    'mmffVariant : “MMFF94” or “MMFF94s”'

    seeded coordinate generation, if = -1, no random seed provided

    removes starting coordinates to ensure reproducibility

    max attempts, to increase if issues are encountered during optimization
    """
    if do_charge:
        property_name = "_GasteigerCharge"

    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    if mol is None:
        err = True
    else:
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail:
            raise ValueError(sanitize_fail)
            err = True

        if do_geometry:
            mol, err = OptGeometry(mol, max_iter, mmffvariant, seed, max_attempts)

        mol = Chem.RemoveHs(mol)

        if do_charge:
            mol, property_name, err = GetCharge(mol, property_name, do_charge)

    if err:
        print("Error in molecule pre-treatment")

    return mol, property_name, err


def GetCharge(mol: Chem.Mol, property_name: str, do_charge: bool):
    # Gasteiger-Marsili Charges
    if do_charge:
        AllChem.ComputeGasteigerCharges(mol)
        property_name = "_GasteigerCharge"
        err = CheckMol(mol, property_name, do_charge)

    # partial charges
    else:
        err = CheckMol(mol, property_name, do_charge)

        if err:
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()
            w = np.ones((n_atoms, 1)) / n_atoms
            w = np.asarray(map(float, w))
            property_name = "equal_w"
            err = False

        else:
            # prepares molecule
            mol = Chem.RemoveHs(mol)
            n_atoms = mol.GetNumAtoms()
            # takes properties
            properties = mol.GetPropsAsDict()
            string_values = properties[
                property_name
            ]  # extracts the property according to the set name
            string_values = string_values.split("\n")
            w = np.asarray(map(float, string_values))

        for atom_ix in range(n_atoms):
            atom: Chem.Atom = mol.GetAtomWithIdx(atom_ix)
            atom.SetDoubleProp(property_name, w[atom_ix])

    return mol, property_name, err


def CheckMol(mol: Chem.Mol, property_name: str, do_charge: bool) -> bool:
    """Checks if the property (as specified by "property_name") is annotated and gives err = 0 if it is"""
    n_atoms = mol.GetNumAtoms()

    if do_charge:
        for atom_ix in range(n_atoms):
            atom: Chem.Atom = mol.GetAtomWithIdx(atom_ix)
            value = atom.GetProp(property_name)
            # checks for error (-nan, inf, nan)
            if value in ["-nan", "nan", "inf"]:
                return True

    else:
        properties = mol.GetPropsAsDict()
        string_values = properties[property_name]  # extracts the property according to the set name
        if string_values == "" or string_values == [""]:
            return True

    if n_atoms < 4:
        return True
    return False


def OptGeometry(
    mol: Chem.Mol, max_iter: int, mmffvariant: int, seed: int, max_attempts: int
) -> Chem.Mol:
    err = False
    try:
        mol = Chem.AddHs(mol)
        a = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            useBasicKnowledge=True,
            randomSeed=seed,
            clearConfs=True,
            maxAttempts=max_attempts,
        )
        if a == -1:
            err = False

        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iter, mmffVariant=mmffvariant)

    except ValueError:
        err = True
    except TypeError:
        err = True

    return mol, err


def PrepareMolFromSDF(
    filename_in,
    do_geometry=True,
    do_charge=False,
    property_name="_GasteigerCharge",
    max_iter=1000,
    mmffvariant="MMFF94",
    seed=26,
    max_attempts=100,
):
    vs_library = Chem.SDMolSupplier(filename_in)
    vs_library_prepared = []

    n_mols = len(vs_library)

    for ix, mol in tqdm(enumerate(vs_library), total=n_mols):
        mol, _, err = PrepareMol(
            mol, do_geometry, do_charge, property_name, max_iter, mmffvariant, seed, max_attempts
        )

        if err:
            tqdm.write(f"Molecule {ix} of {n_mols} not computed.")

        vs_library_prepared.append(mol)

    return vs_library_prepared


def GetCoordinatesAndProps(mol: Chem.Mol, property_name="partial_charges", do_charge=True):
    """Extracts all of the useful chemical information, i.e., the partial charge and the coordinates and formats it
    for atomic centred covariance matrix calculation.
    ====================================================================================================================
    :param
    mol: rdkit molecule
    do_charge: if True, the charges are computed
    do_geom: if True, it calculates MMF 3D coordinates
    :returns
    coords (n_atoms x 3): geometrical matrix (x-y-z coords)
    w (n_atoms x 1): partial charge array
    ====================================================================================================================
    Francesca Grisoni, 05/2018, v. beta
    ETH Zurich
    """
    # molecule preparation
    mol, property_name, err = GetCharge(mol, property_name, do_charge)

    if not err:
        # pre-allocation
        n_atoms = mol.GetNumAtoms()  # num atoms
        coords = np.zeros((n_atoms, 3))  # init coords
        w = np.zeros((n_atoms, 1))  # init weights

        # coordinates and property
        for atom_ix in range(n_atoms):  # loops over atoms, gets 3D coordinate matrix
            # gets atomic positions
            pos = mol.GetConformer().GetAtomPosition(atom_ix)
            coords[atom_ix,] = [pos.x, pos.y, pos.z]

            # gets atomic properties
            atom: Chem.Atom = mol.GetAtomWithIdx(atom_ix)
            value = atom.GetProp(property_name)
            w[atom_ix] = value

        # checks the weight values computed and throws and error if they are all 0
        if all(v == 0 for v in w):
            err = True
    else:
        coords = []
        w = []

    return coords, w, err
