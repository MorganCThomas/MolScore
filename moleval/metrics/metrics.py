"""
Adapted from MOlecular SEtS (MOSES) benchmark
https://github.com/molecularsets/moses
"""

import warnings
import multiprocessing
from collections import Counter
import numpy as np
import rdkit
from scipy.spatial.distance import cosine as cos_distance
from scipy.stats import wasserstein_distance
from scipy import linalg
from rdkit.Chem import DataStructs

try:
    from molbloom import buy
    _has_molbloom = True
except (ImportError, TypeError) as e:
    print(f"Molbloom incompatible, skipping purchasability score: {e}")
    _has_molbloom = False

from moleval.utils import disable_rdkit_log, enable_rdkit_log
from moleval.metrics.fcd_torch import FCD as FCDMetric
from moleval.metrics.metrics_utils import mapper
from moleval.metrics.metrics_utils import SillyWalks
from moleval.metrics.metrics_utils import SA, QED, NP, weight, logP
from moleval.metrics.metrics_utils import compute_fragments, average_agg_tanimoto, \
    compute_scaffolds, fingerprints, numpy_fps_to_bitvectors, sphere_exclusion,\
    get_mol, canonic_smiles, mol_passes_filters, analogues_tanimoto, compute_functional_groups, compute_ring_systems

try:
    from moleval.metrics.posecheck import PoseCheck
    _posecheck_available = True
except Exception as e:
    _posecheck_available = False
    _posecheck_error = e


class GetMetrics(object):
    """
    Computes all available metrics between test (scaffold test)
    and generated sets of SMILES.
    Parameters:
        gen: list of generated SMILES
        n: Chunk size to calculate intermediate statistics
        n_col: Alternatively column name of batch/step variable e.g. "step"
        n_jobs: number of workers for parallel processing
        device: 'cpu' or 'cuda:n', where n is GPU device number
        batch_size: batch size for FCD metric
        pool: optional multiprocessing pool to use for parallelization
        test (None or list): test SMILES. If None, will not compare to test statistics
        test_scaffolds (None or list): scaffold test SMILES. If None, will not compare to
            scaffold test statistics
        ptest (None or dict): precalculated statistics of the test set. If
            None, will not run comparitive statistics. If you specified a custom
            test set, default test statistics will be ignored
        ptest_scaffolds (None or dict): precalculated statistics of the
            scaffold test set If None, will load default scaffold test
            statistics. If you specified a custom test set, default test
            statistics will be ignored
        ptarget (None or dict): precalculated statistics of the target set. If
            None, will not run comparitive statistics
        train (None or list): train SMILES. Only compute novelty as this is usually a very large dataset, to run comparative statistics, submit a sample as test
        target (None or list): target SMILES. If none, will not run comparative statistics
        run_fcd (bool): Whether to compute FCD if pre-statistics aren't supplied
        normalize (bool): Whether to normalize metrics by number of molecules (this may not behave or apply to all metrics like IntDiv, or SNN, RS_<ref> etc.)
        cumulative (bool): Keep a memory of previously generated SMILES and only compute new unique gens, sum or average with previous metrics depending on normalize.
    Available metrics:
        ----- Intrinsic metrics ----
        * # - Number of molecules
        * Validity - Ratio of valid molecules
        * # Valid - Number of valid molecules
        * Uniqueness - Ratio of valid unique molecules
        * # Valid & Unique - Number of valid unique molecules
        * Internal diversity (IntDiv1) - Average average Tanimoto similarity
        * Internal diversity 2 (IntDiv2) - Square root of mean squared Tanimoto similarity
        * Sphere exclusion diversity (SEDiv) - Ratio of diverse molecules in a 1k sub-sample according to sphere exclusion at a Tanimoto distance of 0.65
        * Scaffold diversity (ScaffDiv) - Internal diversity calculate on Bemis-Murcko scaffolds
        * Scaffold uniqueness - Ratio of unique scaffolds within valid unique molecules
        * Functional groups (FG) - Ratio of unique functional groups (Ertl, J. Cheminform (2017) 9:36) within valid unique molecules
        * Ring systems (RS) - Ratio of the unique ring systems within valid unique molecules
        * Filters - Ratio of molecules that pass MOSES filters (MCF & PAINS)
        ----- Extrinsic/relative metrics -----
        * Frechet ChemNet Distance (FCD) - Distance between the final layer of Molecule Net (Preuer et al. J. Chem. Inf. Model. 2018, 58, 9)
        * Novelty - Ratio of valid unique molecules not found within reference dataset
        * AnalogueSimilarity (AnSim) - Ratio of valid unique molecules with a Tanimoto similarity > 0.4 to any in the reference dataset
        * AnalogueCoverage (AnCov) - Ratio of refernce dataset molecules with a Tanimoto similarity > 0.4 to any valid unique molecule
        * Functional groups (FG) - Cosine similarity between the count vector of functional groups
        * Ring systems (RS) - Cosine similarity between the count vector of ring systems
        * Single nearest neighbour (SNN) - Average nearest neighbour similarity of valid unique molecules to the reference dataset
        * Fragment similarity (Frag) - Cosine similarity between the count vector of fragments
        * Scaffold similarity (Scaf) - Cosine similarity between the count vector of scaffolds
        * Properties (logP, NP, SA, QED, weight) - Wasserstain distance between the generated (valid unique) and reference distribution
    """
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, pool=None,
                 test=None, test_scaffolds=None, ptest=None, ptest_scaffolds=None, train=None, ptrain=None,
                 target=None, ptarget=None, target_structure=None, target_ligand=None, run_fcd=True, normalize=True, cumulative=False):
        """
        Prepare to calculate metrics by declaring reference datasets and running pre-statistics
        """
        self.prev_gen = set()
        self.prev_scaff = set()
        self.prev_fg = set()
        self.prev_rs = set()
        self.prev_metrics = {}
        self.prev_n = {'n': 0, 'n_scaff': 0, 'n_fg': 0, 'n_rs': 0}
        self.normalize = normalize
        self.cumulative = cumulative
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self.pool = pool
        self.close_pool = False
        self.test = test
        self.test_scaffolds = test_scaffolds
        self.train = train
        self.target = target
        self.run_fcd = run_fcd
        # Clean up invalid smiles if necessary
        print('Cleaning up reference smiles')
        for att in ['test', 'test_scaffolds', 'target', 'train']:
            if getattr(self, att) is not None:
                setattr(self, att, remove_invalid(getattr(self, att), canonize=True, n_jobs=self.n_jobs))
        # FCD pre-statistics
        self.ptest = ptest
        self.ptest_scaffolds = ptest_scaffolds
        self.ptrain = ptrain
        self.ptarget = ptarget
        # Posecheck target structure
        self.target_structure = target_structure
        self.target_ligand = target_ligand
        # Later defined
        self.kwargs = None
        self.kwargs_fcd = None
        self.test_int = None
        self.test_scaffolds_int = None
        self.target_int = None

        # Compute any pre-statistics if needed.
        disable_rdkit_log()
        if self.pool is None:
            if self.n_jobs != 1:
                self.pool = multiprocessing.get_context("fork").Pool(n_jobs)
                self.close_pool = True
            else:
                self.pool = 1
        self.kwargs = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size}
        self.kwargs_fcd = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size,
                           'canonize': False} # Canonicalized already

        # If test and test_scaffolds provided calculate intermediate statistics
        if self.test is not None:
            print('Computing test pre-statistics')
            self.test_int = compute_intermediate_statistics(self.test, pool=self.pool, run_fcd=self.run_fcd, **self.kwargs)
            self.test_sw = SillyWalks(reference_mols=self.test, n_jobs=self.n_jobs)
            if not self.ptest: self.ptest = self.test_int.get('FCD')
        if self.test_scaffolds is not None:
            print('Computing test scaffold pre-statistics')
            self.test_scaffolds_int = compute_intermediate_statistics(self.test_scaffolds, pool=self.pool, run_fcd=self.run_fcd, **self.kwargs)
            self.test_scaffolds_sw = SillyWalks(reference_mols=self.test_scaffolds, n_jobs=self.n_jobs)
            if not self.ptest_scaffolds: self.ptest_scaffolds = self.test_scaffolds_int.get('FCD')
        if self.target is not None:
            print('Computing target pre-statistics')
            self.target_int = compute_intermediate_statistics(self.target, pool=self.pool, run_fcd=self.run_fcd, **self.kwargs)
            self.target_sw = SillyWalks(reference_mols=self.target, n_jobs=self.n_jobs)
            if not self.ptarget: self.ptarget = self.target_int.get('FCD')

        # Initialize posecheck
        if self.target_structure:
            if _posecheck_available:
                self.posecheck = PoseCheck(n_jobs=self.n_jobs)
                self.posecheck.load_protein_from_pdb(self.target_structure)
            else:
                warnings.warn(f"PoseCheck: currently unavailable due to {_posecheck_error} error")

    def calculate(self, gen=None, calc_valid=False, calc_unique=False, se_k=1000, sp_k=1000, properties=False, return_stats=False):
        """
        Calculate metrics for a generate de novo dataset
        :param gen: List of de novo generate smiles or mols
        :param calc_valid: Return validity ratio
        :param calc_unique: Return unique ratio
        :param se_k: Sub-sample size for sphere exclusion diversity
        :param sp_k: Sub-sample size for Solow Polasky diversity
        :param verbose: Print updates
        """
        
        # Store metrics as a list of dicts
        metrics = []
        def add_metric(key, value, **kwargs):
            kwargs.update({
                'metric': key,
                'value': value,
            })
            metrics.append(kwargs)
        add_metric('#', len(gen))

        gen, mols, n_valid, fraction_valid, fraction_unique = preprocess_gen(gen, prev_gen=self.prev_gen, n_jobs=self.n_jobs)

        # ----- Intrinsic properties -----

        # Add validity
        if calc_valid and self.normalize:
            add_metric('Validity', fraction_valid)
        add_metric('# valid', n_valid)

        # Add Uniqueness
        if calc_unique and self.normalize:
            add_metric('Uniqueness', fraction_unique)
        add_metric('# valid & unique', len(gen))

        # Compute pre-statistics
        mol_fps = fingerprints(mols, self.pool, already_unique=True, fp_type='morgan')
        scaffs = compute_scaffolds(mols, n_jobs=self.n_jobs)
        scaff_gen = list(scaffs.keys())
        fgs = compute_functional_groups(mols, n_jobs=self.n_jobs)
        rss = compute_ring_systems(mols, n_jobs=self.n_jobs)
        if self.cumulative:
            scaff_gen = list(set(scaff_gen) - set(self.prev_scaff))
            fgs = Counter({k: v for k, v in fgs.items() if k not in self.prev_fg})
            rss = Counter({k: v for k, v in rss.items() if k not in self.prev_rs})
        scaff_mols = mapper(self.pool)(get_mol, scaff_gen) 
        
        # Add novelty
        if self.train:
            if self.normalize:
                add_metric('Novelty', novelty(gen, self.train, self.pool))
            add_metric('# novel', novelty(gen, self.train, self.pool, normalize=False))

        # Add internal diversity
        add_metric('IntDiv1', internal_diversity(gen=mol_fps, n_jobs=self.pool, p=1, device=self.device))
        add_metric('IntDiv2', internal_diversity(gen=mol_fps, n_jobs=self.pool, p=2, device=self.device))
        
        # Add sphere exclusion
        if se_k and (len(mols) < se_k):
            warnings.warn(f'Less than {se_k} molecules so SEDiv is non-standard.')
            add_metric('SEDiv', se_diversity(gen=mol_fps, n_jobs=self.pool, normalize=self.normalize))
        if se_k and (len(gen) >= se_k):
            add_metric(f'SEDiv@{se_k/1000:.0f}k', se_diversity(gen=mol_fps, k=se_k, n_jobs=self.pool, normalize=self.normalize))
        
        # Add Solow Polasky diversity
        if sp_k and (len(mols) < sp_k):
            warnings.warn(f'Less than {sp_k} molecules so SPDiv is non-standard.')
            add_metric('SPDiv', sp_diversity(gen=mols, n_jobs=self.pool, normalize=self.normalize))
        if sp_k and (len(mols) >= sp_k):
            add_metric(f'SPDiv@{sp_k/1000:.0f}k', sp_diversity(gen=mols, k=sp_k, n_jobs=self.pool, normalize=self.normalize))
        
        # Add Scaffold diversity
        add_metric('# scaffolds', len(scaff_gen))
        add_metric('ScaffDiv', internal_diversity(gen=scaff_mols, n_jobs=self.pool, p=1, device=self.device, fp_type='morgan'))
        if len(scaff_gen):
            add_metric('ScaffUniqueness', len(scaff_gen)/len(gen))
        else:
            add_metric('ScaffUniqueness', 0.)
        
        # Calculate propertion of unique functional groups (FG) relative to all functional groups
        if sum(fgs.values()):
            fgs_n = sum(fgs.values())
            fgs_value = len(list(fgs.keys()))/fgs_n if self.normalize else len(list(fgs.keys()))
            add_metric('FG', fgs_value, n=fgs_n)
        else:
            add_metric('FG', 0.)
            
        # Calculate number of ring systems (RS) relative to sample size
        if sum(rss.values()):
            rss_n = sum(rss.values())
            rss_value = len(list(rss.keys()))/rss_n if self.normalize else len(list(rss.keys()))
            add_metric('RS', rss_value, n=rss_n)
        else: 
            add_metric('RS', 0.)
        
        # Calculate filters
        add_metric('Filters', fraction_passes_filters(mols, self.pool, normalize=self.normalize))

        # Calculate purchasability
        if _has_molbloom:
            purchasable = mapper(self.pool)(buy, gen)
            add_metric('Purchasable_ZINC20', np.mean(purchasable) if self.normalize else np.sum(purchasable))

        # ---- PoseCheck metrics ----
        if self.target_structure and _posecheck_available:
            self.posecheck.load_ligands_from_mols(mols, add_hs=True)
            clashes = self.posecheck.calculate_clashes()
            add_metric('PC_Clashes', np.nanmean(clashes), dist=clashes)
            strain_energy = self.posecheck.calculate_strain_energy()
            add_metric('PC_StrainEnergy', np.nanmean(strain_energy), dist=strain_energy)
            if self.target_ligand:
                similarities = self.posecheck.calculate_interaction_similarity(
                    ref_lig_path = self.target_ligand,
                    similarity_func = "Tanimoto",
                    count = False
                )
                add_metric('PC_Interactions', np.nanmean(similarities), dist=similarities)

        # ---- Extrinsic properties ----

        # Calculate FCD
        if self.run_fcd:
            pgen = FCDMetric(**self.kwargs_fcd).precalc(gen)
            if self.ptrain:
                add_metric('FCD_train', FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptrain))
            if self.ptest:
                add_metric('FCD_test', FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptest))
            if self.ptest_scaffolds:
                add_metric('FCD_testSF', FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptest_scaffolds))
            if self.ptarget:
                add_metric('FCD_target', FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptarget))

        # Test metrics
        if self.test_int is not None:
            add_metric('Novelty_test', novelty(gen, self.test, self.pool, normalize=self.normalize))
            ansim_test, ancov_test = FingerprintAnaloguesMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_int['Analogue'], normalize=self.normalize)
            add_metric('AnSim_test', ansim_test)
            add_metric('AnCov_test', ancov_test)
            add_metric('FG_test', FGMetric(**self.kwargs)(pgen={'fgs': fgs}, pref=self.test_int['FG']))
            add_metric('RS_test', RSMetric(**self.kwargs)(pgen={'rss': rss}, pref=self.test_int['RS']))
            test_snns = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_int['SNN'])
            add_metric(
                'SNN_test',
                np.mean(test_snns) if self.normalize else np.sum(test_snns),
                dist=list(test_snns)
            )
            add_metric('Frag_test', FragMetric(**self.kwargs)(gen=mols, pref=self.test_int['Frag']))
            add_metric('Scaf_test', ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.test_int['Scaf']))
            outlier_bits_test = self.test_sw.score_mols(mols)
            add_metric(
                'OutlierBits_test',
                np.mean(outlier_bits_test) if self.normalize else np.sum(outlier_bits_test),
                dist=outlier_bits_test)
            if properties:
                for name, func in [('logP', logP),
                                ('NP', NP),
                                ('SA', SA),
                                ('QED', QED),
                                ('Weight', weight)]:
                    add_metric(f'{name}_test', WassersteinMetric(func, **self.kwargs)(gen=mols, pref=self.test_int[name]))

        # Test scaff metrics
        if self.test_scaffolds_int is not None:
            testsf_snns = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_scaffolds_int['SNN'])
            add_metric(
                'SNN_testSF',
                np.mean(testsf_snns) if self.normalize else np.sum(testsf_snns),
                dist=list(testsf_snns)
            )
            add_metric('Frag_testSF', FragMetric(**self.kwargs)(gen=mols, pref=self.test_scaffolds_int['Frag']))
            add_metric('Scaf_testSF', ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.test_scaffolds_int['Scaf']))
            outlier_bits_testsf = self.test_scaffolds_sw.score_mols(mols)
            add_metric(
                'OutlierBits_testSF',
                np.mean(outlier_bits_testsf) if self.normalize else np.sum(outlier_bits_testsf),
                dist=outlier_bits_testsf
            )

        # Target metrics
        if self.target_int is not None:
            add_metric('Novelty_target', novelty(gen, self.target, self.pool, normalize=self.normalize))
            ansim_target, ancov_target = FingerprintAnaloguesMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.target_int['Analogue'], normalize=self.normalize)
            add_metric('AnSim_target', ansim_target)
            add_metric('AnCov_target', ancov_target)
            add_metric('FG_target', FGMetric(**self.kwargs)(pgen={'fgs': fgs}, pref=self.target_int['FG']))
            add_metric('RS_target', RSMetric(**self.kwargs)(pgen={'rss': rss}, pref=self.target_int['RS']))
            target_snns = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.target_int['SNN'])
            add_metric(
                'SNN_target',
                np.mean(target_snns) if self.normalize else np.sum(target_snns),
                dist=list(target_snns)
            )
            add_metric('Frag_target', FragMetric(**self.kwargs)(gen=mols, pref=self.target_int['Frag']))
            add_metric('Scaf_target', ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.target_int['Scaf']))
            outlier_bits_target = self.target_sw.score_mols(mols)
            add_metric(
                'OutlierBits_target',
                np.mean(outlier_bits_target) if self.normalize else np.sum(outlier_bits_target),
                dist=outlier_bits_target
            )
            if properties:
                for name, func in [('logP', logP),
                                ('NP', NP),
                                ('SA', SA),
                                ('QED', QED),
                                ('Weight', weight)]:
                    add_metric(f'{name}_target', WassersteinMetric(func, **self.kwargs)(gen=mols, pref=self.target_int[name]))

        if self.cumulative:
            metrics = self.cumulative_correction(metrics, len(gen), len(scaff_gen), len(list(fgs.keys())), len(list(rss.keys())))
            self.prev_gen.update(gen)
            self.prev_scaff.update(scaff_gen)
            self.prev_fg.update(fgs.keys())
            self.prev_rs.update(rss.keys())
            
        if not return_stats:
            return {m['metric']: m['value'] for m in metrics}
        else:
            return metrics

    def cumulative_correction(self, metrics, n, n_scaffs, n_fgs, n_rss):
        """
        Corrects metrics into a rolling average, such that the final value represents the metric for all molecules generated
        """
        # Skip for first run
        if self.prev_metrics:
            if self.normalize:
                # For most metrics compute weighted sum
                for i, m in enumerate(metrics):
                    key = m['metric']
                    # Keys that require addition
                    if key in ['#', '# valid', '# valid & unique', '# scaffolds']:
                        m['value'] += self.prev_metrics[i]['value']
                    # Keys that require averaging by number of FGs
                    elif key in ['FG']:
                        m['value'] = (m['value'] * n_fgs + self.prev_metrics[i]['value'] * self.prev_n['n_fg']) / (n_fgs + self.prev_n['n_fg'])   
                    # Keys that require averaging by number of RSs
                    elif key in ['RS']:
                        m['value'] = (m['value'] * n_rss + self.prev_metrics[i]['value'] * self.prev_n['n_rs']) / (n_rss + self.prev_n['n_rs'])   
                    # Keys that require averaging by number of scaffolds
                    elif key in ['ScaffDiv']:
                        m['value'] = (m['value'] * n_scaffs + self.prev_metrics[i]['value'] * self.prev_n['n_scaff']) / (n_scaffs + self.prev_n['n_scaff'])
                    # Keys that require averaging by number of molecules
                    else:
                        m['value'] = (m['value'] * n + self.prev_metrics[i]['value'] * self.prev_n['n']) / (n + self.prev_n['n'])
            else:
                # For all metrics sum
                for i, m in enumerate(metrics):
                    m['value'] += self.prev_metrics[i]['value']

        # Update for next run
        self.prev_metrics = metrics
        self.prev_n['n'] += n
        self.prev_n['n_scaff'] += n_scaffs
        self.prev_n['n_fg'] += n_fgs
        self.prev_n['n_rs'] += n_rss
        return metrics

    def property_distributions(self, gen):
        metrics = {}
        if self.test_int is not None:
            for name in ['logP', 'NP', 'SA', 'QED', 'Weight']:
                metrics[f'{name}_test'] = self.test_int[name]['values']
        if self.target_int is not None:
            for name in ['logP', 'NP', 'SA', 'QED', 'Weight']:
                metrics[f'{name}_test'] = self.target_int[name]['values']

        gen = remove_invalid(gen, canonize=True, n_jobs=self.n_jobs)
        gen = list(set(gen))
        mols = mapper(self.pool)(get_mol, gen)
        for name, func in [('logP', logP),
                           ('NP', NP),
                           ('SA', SA),
                           ('QED', QED),
                           ('Weight', weight)]:

            metrics[name] = WassersteinMetric(func, **self.kwargs).precalc(mols)['values']
        return metrics

    def close_pool(self):
        enable_rdkit_log()
        if self.close_pool:
            self.pool.close()
            self.pool.join()
        return


def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None, run_fcd=True):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = multiprocessing.get_context("fork").Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    if run_fcd:
        statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    statistics['Analogue'] = FingerprintAnaloguesMetric(**kwargs).precalc(mols)
    statistics['FG'] = FGMetric(**kwargs).precalc(mols)
    statistics['RS'] = RSMetric(**kwargs).precalc(mols)
    for name, func in [('logP', logP),
                       ('NP', NP),
                       ('SA', SA),
                       ('QED', QED),
                       ('Weight', weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics


def fraction_passes_filters(gen, n_jobs=1, normalize=True):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    if normalize:
        return np.mean(passes)
    else:
        return np.sum(passes)


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan', p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    p: power for averaging: (mean x^p)^(1/p)
    """
    assert isinstance(gen[0], rdkit.Chem.rdchem.Mol) or isinstance(gen[0], np.ndarray)

    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    else:
        gen_fps = gen

    agg = average_agg_tanimoto(gen_fps, gen_fps, agg='mean', device=device, p=p)

    if p != 1:
        return 1 - np.mean((agg) ** (1 / p))
    else:
        return 1 - np.mean(agg)


def se_diversity(gen, k=None, n_jobs=1, fp_type='morgan',
                 dist_threshold=0.65, normalize=True):
    """
    Computes Sphere exclusion diversity i.e. fraction of diverse compounds according to a pre-defined
     Tanimoto distance on the first k molecules.

    :param k:
    :param gen:
    :param n_jobs:
    :param device:
    :param fp_type:
    :param gen_fps:
    :param dist_threshold:
    :param normalize:
    :return:
    """
    assert isinstance(gen[0], rdkit.Chem.rdchem.Mol) or isinstance(gen[0], np.ndarray) or isinstance(gen[0], str)

    if k is not None:
        if len(gen) < k:
            warnings.warn(
                f"Can't compute SEDiv@{k/1000:.0f} "
                f"gen contains only {len(gen)} molecules"
            )
        np.random.seed(123)
        idxs = np.random.choice(list(range(len(gen))), k, replace=False)
        if isinstance(gen[0], (rdkit.Chem.rdchem.Mol, str)): 
            gen = [gen[i] for i in idxs]
        else: 
            gen = gen[idxs]

    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    elif isinstance(gen[0], str):
        gen_mols = mapper(n_jobs)(get_mol, gen)
        gen_fps = fingerprints(gen_mols, fp_type=fp_type, n_jobs=n_jobs)
    else:
        gen_fps = gen

    bvs = numpy_fps_to_bitvectors(gen_fps, n_jobs=n_jobs)
    no_diverse = sphere_exclusion(fps=bvs, dist_thresh=dist_threshold)
    if normalize:
        return no_diverse / len(gen)
    else:
        return no_diverse

def sp_diversity(gen, k=None, n_jobs=1, normalize=True):
    """
    Computes Solow Polasky diversity on the first k molecules.

    :param k:
    :param gen:
    :param n_jobs:
    :param device:
    :param fp_type:
    :param gen_fps:
    :param dist_threshold:
    :param normalize:
    :return:
    """
    assert isinstance(gen[0], rdkit.Chem.rdchem.Mol) or isinstance(gen[0], np.ndarray)

    if k is not None:
        if len(gen) < k:
            warnings.warn(
                f"Can't compute SEDiv@{k/1000:.0f} "
                f"gen contains only {len(gen)} molecules"
            )
        np.random.seed(123)
        idxs = np.random.choice(list(range(len(gen))), k, replace=False)
        if isinstance(gen[0], rdkit.Chem.rdchem.Mol): 
            gen = [gen[i] for i in idxs]
        else: 
            gen = gen[idxs]

    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_fps = fingerprints(gen, fp_type='morgan', n_jobs=n_jobs, morgan__r=3, morgan__n=2048)
    else:
        gen_fps = gen

    bvs = numpy_fps_to_bitvectors(gen_fps, n_jobs=n_jobs)
    # Compute distances
    dist = 1 - np.array([DataStructs.BulkTanimotoSimilarity(f, bvs) for f in bvs])
    # Finds unique rows in arr and return their indices
    arr_ = np.ascontiguousarray(dist).view(np.dtype((np.void, dist.dtype.itemsize * dist.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    dist = dist[idxs, :][:, idxs]
    f_ = np.linalg.inv(np.e ** (-10 * dist))
    if normalize:
        return np.sum(f_) / len(gen)
    else:
        return np.sum(f_)
    
def preprocess_gen(gen, prev_gen=[], canonize=True, n_jobs=1):
    """
    Convert to valid, unique and return mols in one function (save compute redundancy)
    :return: (gen, mols, n_valid, fraction_valid, fraction_unique)
    """
    # Convert to mols
    mols = mapper(n_jobs)(get_mol, gen)
    fraction_invalid = 1 - mols.count(None) / len(mols)
    n_valid = len(mols) - mols.count(None)
    # Remove invalid
    if canonize:
        gen = [x for x in mapper(n_jobs)(canonic_smiles, mols) if x is not None]
    else:
        gen = [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    mols = [mol for mol in mols if mol is not None]
    # Count unique (Note this is fraction of valid SMILES), keeping order
    pc = Counter(prev_gen)
    c = Counter()
    unique_gen = []
    unique_mols = []
    for smi, mol in zip(gen, mols):
        # Add to counter
        pc.update([smi])
        c.update([smi])
        if pc[smi] == 1:
            unique_gen.append(smi)
            unique_mols.append(mol)
    fraction_unique = len(list(c.keys())) / len(gen)
    return unique_gen, unique_mols, n_valid, fraction_invalid, fraction_unique

def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]

def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def novelty(gen, train, n_jobs=1, normalize=True):
    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    else:
        gen_smiles = gen
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    if normalize:
        return len(gen_smiles_set - train_set) / len(gen_smiles_set)
    else:
        return len(gen_smiles_set - train_set)


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None, **kwargs):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen, **kwargs)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        snns = average_agg_tanimoto(pref['fps'], pgen['fps'], device=self.device)
        return snns
    


class FingerprintAnaloguesMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen, normalize=True):
        gen_ans, ref_ans = analogues_tanimoto(pref['fps'], pgen['fps'], device=self.device)  # Tuple of bool arrays returned (Analogues, Coverage)
        if normalize:
            return gen_ans.mean(), ref_ans.mean()
        else:
            return gen_ans.sum(), ref_ans.sum()


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class FGMetric(Metric):
    def precalc(self, mols):
        return {'fgs': compute_functional_groups(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['fgs'], pgen['fgs'])


class RSMetric(Metric):
    def precalc(self, mols):
        return {'rss': compute_ring_systems(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['rss'], pgen['rss'])


class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(
            pref['values'], pgen['values']
        )
