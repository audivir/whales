"""# Contains all the necessary code to prepare the molecule:
#   - molecule sanitization (check in "import_prepare_mol"
#     to change advanced sanitiization settings")
#   - geometry optimization (if specified by "do_geom = True"),
#     with the specified settings
"""

# pylint: disable=consider-using-assignment-expr
from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Atom, Draw, Mol, rdmolops
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def prepare_mol_from_sdf(
    filename_in: str,
    do_geometry: bool = True,
    do_charge: bool = False,
    property_name: str = "_GasteigerCharge",
    max_iter: int = 1000,
    mmffvariant: str = "MMFF94",
    seed: int = 26,
    max_attempts: int = 100,
) -> list[Mol | None]:
    """Prepares a molecule from an SDF file, with the specified settings.
    Returns a list of prepared molecules.
    """
    vs_library = Chem.SDMolSupplier(filename_in)
    vs_library_iter: Iterator[Mol | None] = iter(vs_library)
    vs_library_prepared: list[Mol | None] = []

    nmol = len(vs_library)

    for ix, mol in enumerate(vs_library_iter):
        if ix % 50 == 0:
            print(f"Molecule: {ix}")

        prep_mol = prepare_mol(
            mol, do_geometry, do_charge, property_name, max_iter, mmffvariant, seed, max_attempts
        )

        if not prep_mol:
            print(f"Molecule {ix} of {nmol} not computed.")

        vs_library_prepared.append(prep_mol)
    return vs_library_prepared


def prepare_mol(
    mol: Mol | None,
    do_geometry: bool = True,
    do_charge: bool = True,
    property_name: str = "_GasteigerCharge",
    max_iter: int = 1000,
    mmffvariant: str = "MMFF94",
    seed: int = 26,
    max_attempts: int = 5,
) -> Mol | None:
    """# 'mmffVariant : “MMFF94” or “MMFF94s”'
    # seeded coordinate generation, if = -1, no random seed provided
    # removes starting coordinates to ensure reproducibility
    # max attempts, to increase if issues are encountered during optimization
    """
    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    opt_err = False
    charge_err = False

    # sanitization
    if mol is None:
        return None

    # sanitize
    sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
    if sanitize_fail:  # type: ignore[truthy-bool]
        raise ValueError(sanitize_fail)
        # err = 1

    if do_geometry:
        mol, opt_err = opt_geometry(mol, max_iter, mmffvariant, seed, max_attempts)

    # calculates or assigns atom charges based on what annotated in do_charge
    mol = rdmolops.RemoveHs(mol)

    if do_charge:
        property_name = "_GasteigerCharge"
        mol, _, charge_err = get_charge(mol, property_name, do_charge)

    if opt_err or charge_err:
        print("Error in molecule pre-treatment")
        return None

    return mol


def opt_geometry(
    mol: Mol, max_iter: int, mmffvariant: str, seed: int, max_attempts: int
) -> tuple[Mol, bool]:
    """Optimizes the geometry of a molecule, with the specified settings."""
    try:
        mol = rdmolops.AddHs(mol)
        conf_id = AllChem.EmbedMolecule(  # type: ignore[attr-defined]
            mol,
            useRandomCoords=True,
            useBasicKnowledge=True,
            randomSeed=seed,
            clearConfs=True,
            maxAttempts=max_attempts,
        )

        AllChem.MMFFOptimizeMolecule(  # type: ignore[attr-defined]
            mol, maxIters=max_iter, mmffVariant=mmffvariant, confId=conf_id
        )

        return mol, False

    except (ValueError, RuntimeError):
        return mol, True


def get_charge(mol: Mol, property_name: str, do_charge: bool) -> tuple[Mol, str, bool]:
    err = False

    # partial charges
    if not do_charge:
        err = check_mol(mol, property_name, do_charge)
        if not err:
            # prepares molecule
            mol = Chem.RemoveHs(mol)
            n_at = mol.GetNumAtoms()
            # takes properties
            list_prop = mol.GetPropsAsDict()
            # extracts the property according to the set name
            string_values = list_prop[property_name]
            string_values = string_values.split("\n")
            w = np.asarray([float(i) for i in string_values])

        else:
            mol = Chem.AddHs(mol)
            n_at = mol.GetNumAtoms()
            w = np.ones((n_at, 1)) / n_at
            w = np.asarray([float(i) for i in w])  # same format as previous calculation
            property_name = "equal_w"
            err = False

        # extract properties
        for atom in range(n_at):
            atom_obj: Atom = mol.GetAtomWithIdx(atom)
            atom_obj.SetDoubleProp(property_name, w[atom])

        mol = Chem.RemoveHs(mol)

    # Gasteiger-Marsili Charges
    else:
        AllChem.ComputeGasteigerCharges(mol)  # type: ignore[attr-defined]
        err = check_mol(mol, property_name, do_charge)

    return mol, property_name, err


# -----------------------------------------------------------------------------
def check_mol(mol: Mol, property_name: str, do_charge: bool) -> bool:
    """Checks if the property (as specified by "property_name")
    is annotated and gives err = False if it is
    """
    n_at = mol.GetNumAtoms()
    if do_charge is False:
        list_prop = mol.GetPropsAsDict()
        string_values = list_prop[property_name]  # extracts the property according to the set name
        if string_values in ("", [""]):
            return True

    else:
        atom = 0
        while atom < n_at:
            atom_obj: Atom = mol.GetAtomWithIdx(atom)
            value = atom_obj.GetProp(property_name)
            # checks for error (-nan, inf, nan)
            if value in {"-nan", "nan", "inf"}:
                return True

            atom += 1

    # checks for the number of atoms
    return n_at < 4  # noqa: PLR2004


# -----------------------------------------------------------------------------
def do_map(
    mol: Mol,
    fig_name: str | None = None,
    lab_atom: bool = False,
    text: bool = False,
    # MapMin: int = 0, # not used
    # MapMax: int = 1, # not used
) -> None:
    # settings

    scale = -1  # size of dots
    coordscale = 1  # coordinate scaling
    colmap = "bwr"

    mol, _, err = get_charge(mol, property_name="_GasteigerCharge", do_charge=True)

    if err == 1:
        print("Error in charge calculation")

    contribs = [
        mol.GetAtomWithIdx(ix).GetDoubleProp("_GasteigerCharge") for ix in range(mol.GetNumAtoms())
    ]

    drawer = Draw.MolDraw2DCairo(400, 400)  # or MolDraw2DCairo
    drawer.drawOptions().clearBackground = True  # type: ignore[assignment]
    drawer.drawOptions().setBackgroundColour((1, 1, 1))  # white background

    SimilarityMaps.GetSimilarityMapFromWeights(  # type: ignore[no-untyped-call]
        mol,
        contribs,
        drawer,
        coordScale=coordscale,
        colorMap=colmap,
        colors="w",
        alpha=0,
        scale=scale,
    )
    drawer.FinishDrawing()

    plt.imshow(Image.open(BytesIO(drawer.GetDrawingText())))
    fig = plt.gcf()

    # SimilarityMaps.Draw.MolDrawOptions.clearBackground
    if lab_atom is False:
        for elem in fig.axes[0].get_children():
            if isinstance(elem, matplotlib.text.Text):
                elem.set_visible(False)

    plt.axis("off")

    if text is True:
        for at in range(mol.GetNumAtoms()):
            x = mol._atomPs[at][0]  # pylint: disable=protected-access
            y = mol._atomPs[at][1]  # pylint: disable=protected-access
            plt.text(
                x,
                y,
                f"{contribs[at]:.2f}",
                path_effects=[PathEffects.withStroke(linewidth=1, foreground="blue")],
            )

    if fig_name is not None:
        fig.savefig(fig_name, bbox_inches="tight")

    plt.show()


def frequent_scaffolds(
    suppl: Sequence[Mol], output_type: str = "supplier"
) -> list[tuple[str, int]] | list[Mol]:
    """Starting from a supplier file, the function computes the most
    frequently recurring scaffolds and returns them as a
    supplier file (if output_type='supplier') or as a counter file.
    """
    scaff_list: list[str] = []

    for mol in suppl:
        scaf_smi: str = MurckoScaffoldSmiles(mol=mol)  # type: ignore[no-untyped-call]
        scaff_list.append(scaf_smi)

    freq_scaffolds_counter = Counter(scaff_list)

    freq_scaffolds = freq_scaffolds_counter.most_common()

    if output_type == "supplier":
        # converts it back in a supplier file,
        suppl_new: list[Mol] = []
        for scaf_smi, count in freq_scaffolds:
            mol = Chem.MolFromSmiles(scaf_smi)
            name = str(round((count / len(suppl)) * 100, 2)) + "%"

            # assigns the molecule name as the percentage occurrence
            mol.SetProp("_Name", name)
            suppl_new.append(mol)

        return suppl_new

    return freq_scaffolds
