import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict

###############################################################################
def GetMFPidx(rootbondlist,dist,selectedAtom=[],bondIDs=[],endatoms=[]):
    if not rootbondlist:
        return bondIDs,endatoms
    newbondlist = []
    newendatom = []
    if bondIDs:
        newbondID = bondIDs[-1].copy()
    else:
        newbondID = []
    for root,bonds in rootbondlist:
        for bond in bonds:
            bondidx = bond.GetIdx()
            newroot = bond.GetOtherAtom(root)
            if newroot.GetIdx() in selectedAtom:
                newbondID.append(bond.GetIdx())
                continue
            newbondID.append(bond.GetIdx())
            newendatom.append(newroot.GetIdx())
            newbonds = []
            for newbond in newroot.GetBonds():
                otheratom = newbond.GetOtherAtom(newroot)
                if otheratom.GetIdx() != root.GetIdx():
                    newbonds.append(newbond)
            newbondlist.append((newroot,newbonds))
    selectedAtom+=newendatom
    bondIDs.append(list(set(newbondID)))
    endatoms.append(newendatom)
    if dist == 0:
        return bondIDs,endatoms
    return GetMFPidx(newbondlist,dist-1,selectedAtom,bondIDs,endatoms)
fakeatom = Chem.Atom(0)
def GetMFP(mol,maxR=2):
    MFP = defaultdict(int)
    for i in range(mol.GetNumAtoms()):
        root = mol.GetAtomWithIdx(i)
        if len(root.GetBonds()) == 0: # When it's just an atom
            tmol = Chem.RWMol(mol.__copy__())
            for j in reversed(range(tmol.GetNumAtoms())):
                if j != i:
                    tmol.RemoveAtom(j)
            for atom in tmol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
            MFP[Chem.MolToSmiles(tmol)] += 1
            continue
        bondids,endatoms = GetMFPidx([(root,root.GetBonds())],maxR,[],[],[])
        for bondid,endatom in zip(bondids,endatoms):
            atommap = {}
            submol = Chem.RWMol(Chem.PathToSubmol(mol,bondid,atomMap=atommap))
            for atom in submol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
            for idx in endatom:
                submol.ReplaceAtom(atommap[idx],fakeatom)
            rootatom = submol.GetAtomWithIdx(atommap[i])
            #rootatom.SetProp('smilesSymbol',rootatom.GetSymbol())
            MFP[Chem.MolToSmiles(submol,rootedAtAtom=atommap[i])]+=1
    return MFP
###############################################################################
'''
smiles = ['CC','CCC','C=C','CC(C)C','C=C(C)C','C1CC1']
radius = 5

FPs = []
for s in smiles:
    mol= Chem.MolFromSmiles(s)
    FP = GetMFP(mol,radius)
    FPs.append(FP)
    print(FP)
    
import numpy as np
uniqueFPs = []
for fps in FPs:
    for fp in fps:
        if fp not in uniqueFPs:
            uniqueFPs.append(fp)
            
X = np.zeros((len(smiles),len(uniqueFPs)))
for i,fps in enumerate(FPs):
    for fp in fps:
        j = uniqueFPs.index(fp)
        X[i,j] = fps[fp]
X[X>0]=1
print(X)

from rdkit.Chem import Draw
Draw.MolToImage(Chem.MolFromSmiles(uniqueFPs[5]))
'''
