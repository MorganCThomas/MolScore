import logging
import os
from functools import partial
from typing import Tuple, Union

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from molscore.scoring_functions.utils import Pool, SimilarityMeasures

logger = logging.getLogger("align3d")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class Align3D:
    """Align molecules in 3D using open-source 3D align via RDKit, atleast one reference molecule must be specified."""

    return_metrics = ["O3A_score", "shape_sim", "pharm_sim", "shape_score"]

    def __init__(
        self,
        prefix: str,
        ref_smiles: list = None,
        ref_sdf: Union[str, os.PathLike] = None,
        similarity_method: str = "Tanimoto",
        agg_method: str = "mean",
        max_confs: int = 100,
        pharmacophore_similarity: bool = True,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: Reference SMILES for alignment, these will be aligned to ref_sdf or aligned together
        :param ref_sdf: Load references from an sdf file for alignment
        :param similarity_method: Method for comparing similarity between between 3D molecules [Tanimoto, Tversky]
        :param agg_method: Method for aggregating score accross several reference molecules [mean, median, max, min]
        :param max_confs: Maximum number of conformers to generate when assessing alignments
        :param pharmacophore_similarity: Whether to additionally calculate a pharamcophore fingerprint that will be added to the similarity measure
        :param n_jobs: Number of jobs for parallelization
        """
        self.prefix = prefix.strip().replace(" ", "_")
        self.ref_smiles = ref_smiles if ref_smiles is not None else []
        self.ref_sdf = ref_sdf
        self.max_confs = max_confs
        assert similarity_method in ["Tanimoto", "Tversky"]
        self.similarity_method = similarity_method
        assert agg_method in ["mean", "median", "max", "min"]
        self.agg_method = getattr(np, agg_method)
        self.pharmacophore_similarity = pharmacophore_similarity
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.ref_mols = []

        # Check input
        assert (len(self.ref_smiles) > 0) or (
            self.ref_sdf is not None
        ), "Must specify reference smiles or SDF file"
        if self.ref_sdf:
            assert os.path.exists(ref_sdf), "SDF file not found"

        # Parse sdf
        if self.ref_sdf:
            # Read in
            to_align = []
            supplier = Chem.SDMolSupplier(self.ref_sdf)
            for i, mol in enumerate(supplier):
                if mol:
                    if mol.GetNumConformers() == 1:
                        mol = Chem.AddHs(mol, addCoords=True)
                        self.ref_mols.append(mol)
                    else:
                        logger.warning(
                            f"Molecule {i} from {os.path.basename(self.ref_sdf)} has no conformation"
                        )
                        mol = self.prepare_mol(mol)
                        to_align.append(mol)
                else:
                    logger.warning(
                        f"Failed to load molecule {i} from {os.path.basename(self.ref_sdf)}"
                    )

            # Align
            if len(self.ref_mols) > 0:
                # Align to reference molecules
                for mol in to_align:
                    aligned_mol = self.align_to_ref_mols(mol)
                    self.ref_mols.append(aligned_mol)
            else:
                # Align to each other
                logger.warning(
                    "Optimizing starting conformation, this may take several minutes"
                )
                # Optimize starting conformation
                mol, i = self.optimize_starting_conf(to_align)
                to_align.pop(i)
                self.ref_mols.append(mol)
                # Align everything else
                for mol in to_align:
                    aligned_mol = self.align_to_ref_mols(mol)
                    self.ref_mols.append(aligned_mol)

        # Prepare SMILES
        if self.ref_smiles:
            # Align
            if len(self.ref_mols) > 0:
                # Align to reference molecules
                for smi in self.ref_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mol = self.prepare_mol(mol)
                        aligned_mol = self.align_to_ref_mols(mol)
                        self.ref_mols.append(aligned_mol)
                    else:
                        logger.warning(f"SMILES parsing failed: {smi}")
            else:
                # Align to each other
                logger.warning(
                    "Optimizing starting conformation, this may take several minutes"
                )
                # Prepare all
                to_align = []
                for smi in self.ref_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mol = self.prepare_mol(mol)
                        to_align.append(mol)
                    else:
                        logger.warning(f"SMILES parsing failed: {smi}")
                # Optimize starting conformation
                mol, i = self.optimize_starting_conf(to_align)
                to_align.pop(i)
                self.ref_mols.append(mol)
                # Align everything else
                for mol in to_align:
                    aligned_mol = self.align_to_ref_mols(mol)
                    self.ref_mols.append(aligned_mol)

        # Calculate fps
        self.ref_fps = [self.get_3D_pharmacophore_fp(mol) for mol in self.ref_mols]

        assert (
            len(self.ref_mols) > 0
        ), "Zero reference molecules due to processing errors"

    def optimize_starting_conf(self, mols):
        """Take first valid molecule and select conformation with best average alignment to other mols."""
        # Optimize starting conformation
        for i, mol in enumerate(mols):
            if mol:
                conf_scores = []
                for conf_id in range(mol.GetNumConformers()):
                    omol_scores = []
                    for omol in mols:
                        # Skip itself
                        if mol == omol:
                            continue
                        # Find best other molecule conf for alignment
                        _, (_, score) = self.align_best_conf(
                            mol=omol, ref_mol=mol, ref_conf_id=conf_id
                        )
                        omol_scores.append(score)
                    # Add average alignment score for this conf to other reference molecules
                    conf_scores.append(np.mean(omol_scores))
                # Set conformer as best average alignment
                mol = self.set_conformer(mol, conf_id=int(np.argmax(conf_scores)))
                break
            else:
                continue

        return mol, i

    def align_to_ref_mols(self, mol):
        """Align mol to reference molecules by selecting closes alignment."""
        # Get best conf
        best_confs = [self.align_best_conf(mol, ref_mol) for ref_mol in self.ref_mols]
        # Align to closest only
        best_conf = sorted(best_confs, key=lambda x: x[1][1], reverse=True)[0]
        mol, (best_conf_id, score) = best_conf
        # Set conf
        mol = self.set_conformer(mol, best_conf_id)
        return mol

    @staticmethod
    def prepare_mol(mol: Chem.rdchem.Mol, max_confs: int = 100) -> Chem.rdchem.Mol:
        """Prepare a mol for alignment by add hydrogens and generating conformations."""
        # Add Hs
        mol = Chem.AddHs(mol)
        # Generate confs
        Chem.EmbedMultipleConfs(mol, numConfs=max_confs, pruneRmsThresh=0.1)
        # Minimize
        # Chem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        # Chem.MMFFOptimizeMoleculeConfs(mol,  mmffVariant='MMFF94s')
        return mol

    @staticmethod
    def align_best_conf(
        mol: Chem.rdchem.Mol, ref_mol: Chem.rdchem.Mol, ref_conf_id: int = -1
    ) -> Tuple[Chem.rdchem.Mol, Tuple[int, float]]:
        """Return the conf_id and score (higher is better) of the best conformer to align to the ref."""
        # Align all confs
        res = Chem.GetO3AForProbeConfs(
            prbMol=mol, refMol=ref_mol, refCid=ref_conf_id, maxIters=100
        )
        # Add index
        res = list(enumerate(res))
        # Sort by highest score
        res = sorted(res, key=lambda x: x[1].Score(), reverse=True)
        # Align best conf
        res[0][1].Align()
        return mol, (res[0][0], res[0][1].Score())  # mol, (conf_id, score)

    @staticmethod
    def align_joint_best_conf(
        mol: Chem.rdchem.Mol, ref_mol: Chem.rdchem.Mol
    ) -> Tuple[int, int, float]:
        """Return the conf_id's and rmsd of the best conformers between a probe and ref."""
        best_score = (-1, -1, 0)
        for ref_conf_id in range(ref_mol.GetNumConformers()):
            res = Chem.GetO3AForProbeConfs(
                prbMol=mol, refMol=ref_mol, refCid=ref_conf_id, maxIters=100
            )
            res_score = int(np.max([res.Score() for r in res]))
            prb_conf_id = int(np.argmax([res.Score() for r in res]))
            if res_score > best_score[2]:
                best_score = (prb_conf_id, ref_conf_id, res_score)
        # Re-do alignment for best indexes
        res = Chem.GetO3A(
            prbMol=mol,
            refMol=ref_mol,
            prbCid=best_score[0],
            refCid=best_score[1],
            maxIters=100,
        )
        res.Align()
        return (
            mol,
            ref_mol,
            best_score,
        )  # mol, ref_mol, (prb_conf_id, ref_conf_id, score)

    @staticmethod
    def set_conformer(mol: Chem.rdchem.Mol, conf_id: int):
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
        return new_mol

    @staticmethod
    def get_shape_similarity(mol1, mol2, mol1_cid=-1, mol2_cid=-1, method="Tanimoto"):
        if method == "Tversky":
            sim = Chem.ShapeTverskyIndex(
                mol1, mol2, alpha=0.5, beta=0.5, confId1=mol1_cid, confId2=mol2_cid
            )
        else:
            sim = 1 - Chem.ShapeTanimotoDist(
                mol1, mol2, confId1=mol1_cid, confId2=mol2_cid
            )
        return sim

    @staticmethod
    def get_3D_pharmacophore_fp(mol: Chem.rdchem.Mol, conf_id=-1):
        fp = Generate.Gen2DFingerprint(
            mol,
            Gobbi_Pharm2D.factory,
            dMat=Chem.Get3DDistanceMatrix(mol, confId=conf_id),
        )
        return fp

    @staticmethod
    def get_pharmacophore_similarity(fp1, fp2):
        measure = SimilarityMeasures.get("Tanimoto")
        return measure(fp1, fp2)

    @staticmethod
    def score_smi(
        smi: str,
        prefix: str,
        ref_mols: list,
        ref_fps: list,
        max_confs: int,
        similarity_method: str,
        agg_method,
        pharmacophore: bool,
    ):
        result = {"smiles": smi}
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Prepare confs
            mol = Align3D.prepare_mol(mol, max_confs=max_confs)
            # Align to each ref mol
            O3A_scores = []
            cids = []
            sim_scores = []
            fsim_scores = []
            for rmol, rfp in zip(ref_mols, ref_fps):
                mol, (cid, score) = Align3D.align_best_conf(mol, ref_mol=rmol)
                O3A_scores.append(score)
                cids.append(cid)
                sim_scores.append(
                    Align3D.get_shape_similarity(
                        mol1=mol, mol2=rmol, mol1_cid=cid, method=similarity_method
                    )
                )
                fsim_scores.append(
                    Align3D.get_pharmacophore_similarity(
                        Align3D.get_3D_pharmacophore_fp(mol, conf_id=cid), rfp
                    )
                )
            # Align mol to most similar rfp
            best_ref_idx = int(np.argmax(O3A_scores))
            best_cid_idx = cids[best_ref_idx]
            algn = Chem.GetO3A(
                prbMol=mol,
                refMol=ref_mols[best_ref_idx],
                maxIters=100,
                prbCid=best_cid_idx,
            )
            algn.Align()
            mol = Align3D.set_conformer(mol, conf_id=best_cid_idx)
            # Aggregate scores
            result.update(
                {
                    f"{prefix}_mol": mol,
                    f"{prefix}_O3A_score": agg_method(O3A_scores),
                    f"{prefix}_shape_sim": agg_method(sim_scores),
                    f"{prefix}_pharm_sim": agg_method(fsim_scores),
                }
            )
            if pharmacophore:
                result.update(
                    {
                        f"{prefix}_shape_score": agg_method(sim_scores)
                        + agg_method(fsim_scores)
                    }
                )
            else:
                result.update({f"{prefix}_shape_score": agg_method(sim_scores)})
        else:
            result.update({f"{prefix}_{m}": 0.0 for m in Align3D.return_metrics})
        return result

    def score(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on 3D shape similarity to reference molecules.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Create directory
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_Align3D", step
        )
        os.makedirs(directory, exist_ok=True)
        # Prepare function for parallelization
        pfunc = partial(
            self.score_smi,
            prefix=self.prefix,
            ref_mols=self.ref_mols,
            ref_fps=self.ref_fps,
            max_confs=self.max_confs,
            similarity_method=self.similarity_method,
            agg_method=self.agg_method,
            pharmacophore=self.pharmacophore_similarity,
        )
        # Score individual smiles
        results = [r for r in self.mapper(pfunc, smiles)]
        # Save mols
        for r, name in zip(results, file_names):
            file_path = os.path.join(directory, name + ".sdf")
            try:
                pass
                mol = r.pop(f"{self.prefix}_mol")
                mol.SetProp("_Name", str(name))
                for m in self.return_metrics:
                    mol.SetProp(m, str(r[f"{self.prefix}_{m}"]))
                w = Chem.SDWriter(file_path)
                w.write(mol)
                w.flush()
                w.close()
            except KeyError:
                continue
        return results

    def __call__(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on 3D shape similarity to reference molecules.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles, directory=directory, file_names=file_names)
