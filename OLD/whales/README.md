# WHALES
## Usage
```
mol = Chem.MolFromSmiles('c1ccccc1')
whales.Compute2DCoords(mol)
mol = whales.PrepareMol(mol)
fp = whales.GetFingerprint(mol)
fps = [whales.GetFingerprint(mol) for mol in mols]
fps = whales.StackFingerprints(fps)
```
