"""
Adapted from MOlecular SEtS (MOSES) benchmark
https://github.com/molecularsets/moses
"""

import warnings
from multiprocessing import Pool
import numpy as np
import rdkit
from scipy.spatial.distance import cosine as cos_distance
from fcd_torch import FCD as FCDMetric
from scipy.stats import wasserstein_distance

from moleval.metrics.utils import mapper
from moleval.metrics.utils import disable_rdkit_log, enable_rdkit_log
from moleval.metrics.metrics_utils import SA, QED, NP, weight, logP
from moleval.metrics.metrics_utils import compute_fragments, average_agg_tanimoto, \
    compute_scaffolds, fingerprints, numpy_fps_to_bitvectors, sphere_exclusion,\
    get_mol, canonic_smiles, mol_passes_filters, analogues_tanimoto, compute_functional_groups, compute_ring_systems


# Modify this function so that it doesn't depend on metrics dataset
class GetMosesMetrics(object):
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
            train (None or list): train SMILES. If None, will not run comparative statistics
            target (None or list): target SMILES. If none, will not run comparative statistics
        Available metrics:
            * %valid # Tracked by molscore and so unneccesary
            * Frechet ChemNet Distance (FCD)
            * Fragment similarity (Frag)
            * Scaffold similarity (Scaf)
            * Similarity to nearest neighbour (SNN)
            * Internal diversity (IntDiv)
            * Internal diversity 2: using square root of mean squared
                Tanimoto similarity (IntDiv2)
            * %passes filters (Filters)
            * Distribution difference for logP, SA, QED, weight
            * Novelty (molecules not present in train)
        """
    # TODO add KL divergence?
    # TODO FG / RS inside / outside training data / reference dataset?
    #  https://chemrxiv.org/articles/preprint/Comparative_Study_of_Deep_Generative_Models_on_Chemical_Space_Coverage/13234289
    #  + most common scaff, unique_fg, unique_rs? train_fg_recoverd, test_fg_recovered, test etc.

    def __init__(self, n_jobs=1, device='cpu', batch_size=512, pool=None,
                 test=None, test_scaffolds=None, ptest=None, ptest_scaffolds=None, train=None, ptrain=None,
                 target=None, ptarget=None):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self.pool = pool
        self.close_pool = False
        self.test = test
        self.test_scaffolds = test_scaffolds
        self.train = train
        self.target = target
        # Clean up if necessary
        print('Cleaning up reference smiles')
        for att in ['test', 'test_scaffolds', 'target', 'train']:
            if getattr(self, att) is not None:
                setattr(self, att, remove_invalid(getattr(self, att), canonize=True, n_jobs=self.n_jobs))
        # FCD pre-statistics
        self.ptest = ptest
        self.ptest_scaffolds = ptest_scaffolds
        self.ptrain = ptrain
        self.ptarget = ptarget
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
                self.pool = Pool(n_jobs)
                self.close_pool = True
            else:
                self.pool = 1
        self.kwargs = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size}
        self.kwargs_fcd = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size,
                           'canonize': False}

        # If test and test_scaffolds provided calculate intermediate statistics
        if self.test is not None:
            print('Computing test pre-statistics')
            self.test_int = compute_intermediate_statistics(self.test, n_jobs=self.n_jobs,
                                                            device=self.device, batch_size=self.batch_size,
                                                            pool=self.pool)
        if self.test_scaffolds is not None:
            print('Computing test scaffold pre-statistics')
            self.test_scaffolds_int = compute_intermediate_statistics(self.test_scaffolds, n_jobs=self.n_jobs,
                                                                      device=self.device, batch_size=self.batch_size,
                                                                      pool=self.pool)
        if self.target is not None:
            print('Computing target pre-statistics')
            self.target_int = compute_intermediate_statistics(self.target, n_jobs=self.n_jobs,
                                                              device=self.device, batch_size=self.batch_size,
                                                              pool=self.pool)

    def calculate(self, gen, calc_valid=False, calc_unique=False, unique_k=None, se_k=1000, verbose=False):
        metrics = {}
        metrics['#'] = len(gen)

        # Calculate validity
        if verbose: print("Calculating Validity")
        if calc_valid:
            metrics['Validity'] = fraction_valid(gen, self.pool)

        gen = remove_invalid(gen, canonize=True, n_jobs=self.n_jobs)
        #mols = mapper(self.pool)(get_mol, gen)
        metrics['# valid'] = len(gen)

        # Calculate Uniqueness
        if verbose: print("Calculating Uniqueness")
        if calc_unique:
            metrics['Uniqueness'] = fraction_unique(gen=gen, k=None, n_jobs=self.pool)
            if unique_k is not None:
                metrics[f'Unique@{unique_k/1000:.0f}k'] = fraction_unique(gen=gen, k=unique_k, n_jobs=self.pool)

        # Now subset only unique molecules
        if verbose: print("Computing pre-statistics")
        gen = list(set(gen))
        mols = mapper(self.pool)(get_mol, gen)
        # Precalculate some things
        mol_fps = fingerprints(mols, self.pool, already_unique=True, fp_type='morgan')
        scaffs = compute_scaffolds(mols, n_jobs=self.n_jobs)
        scaff_gen = list(scaffs.keys())
        fgs = compute_functional_groups(mols, n_jobs=self.n_jobs)
        rss = compute_ring_systems(mols, n_jobs=self.n_jobs)
        scaff_mols = mapper(self.pool)(get_mol, scaff_gen)
        metrics['# valid & unique'] = len(gen)

        # Calculate diversity related metrics
        if verbose: print("Calculating Novelty")
        if self.train is not None:
            metrics['Novelty'] = novelty(gen, self.train, self.pool)
        if verbose: print("Calculating Diversity")
        metrics['IntDiv1'] = internal_diversity(gen=mol_fps, n_jobs=self.pool, device=self.device)
        metrics['IntDiv2'] = internal_diversity(gen=mol_fps, n_jobs=self.pool, device=self.device, p=2)
        metrics['SEDiv'] = se_diversity(gen=mols, n_jobs=self.pool)
        if (se_k is not None) and (len(gen) >= se_k):
            metrics[f'SEDiv@{se_k/1000:.0f}k'] = se_diversity(gen=mols, k=se_k, n_jobs=self.pool, normalize=True)
        metrics['ScaffDiv'] = internal_diversity(gen=scaff_mols, n_jobs=self.pool, device=self.device,
                                                 fp_type='morgan')
        metrics['Scaff uniqueness'] = len(scaff_gen)/len(gen)
        # Calculate number of FG and RS relative to sample size
        metrics['FG'] = len(list(fgs.keys()))/len(gen)
        metrics['RS'] = len(list(rss.keys()))/len(gen)
        # Calculate % pass filters
        if verbose: print("Calculating Filters")
        metrics['Filters'] = fraction_passes_filters(mols, self.pool)

        # Calculate FCD
        if verbose: print("Calculating FCD")
        pgen = FCDMetric(**self.kwargs_fcd).precalc(gen)
        if self.ptrain:
            metrics['FCD_train'] = FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptrain)
        if self.ptest:
            metrics['FCD_test'] = FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptest)
        if self.ptest_scaffolds:
            metrics['FCD_testSF'] = FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptest_scaffolds)
        if self.ptarget:
            metrics['FCD_target'] = FCDMetric(**self.kwargs_fcd)(pgen=pgen, pref=self.ptarget)

        # Test metrics
        if self.test_int is not None:
            if verbose: print("Calculating Test metrics")
            metrics['Novelty_test'] = novelty(gen, self.test, self.pool)
            metrics['AnalogueSimilarity_test'], metrics['AnalogueCoverage_test'] = \
                FingerprintAnaloguesMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_int['Analogue'])
            metrics['FG_test'] = FGMetric(**self.kwargs)(pgen={'fgs': fgs}, pref=self.test_int['FG'])
            metrics['RS_test'] = RSMetric(**self.kwargs)(pgen={'rss': rss}, pref=self.test_int['RS'])
            metrics['SNN_test'] = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_int['SNN'])
            metrics['Frag_test'] = FragMetric(**self.kwargs)(gen=mols, pref=self.test_int['Frag'])
            metrics['Scaf_test'] = ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.test_int['Scaf'])
            for name, func in [('logP', logP),
                               ('NP', NP),
                               ('SA', SA),
                               ('QED', QED),
                               ('weight', weight)]:
                metrics[f'{name}_test'] = WassersteinMetric(func, **self.kwargs)(gen=mols, pref=self.test_int[name])

        # Test scaff metrics
        if self.test_scaffolds_int is not None:
            if verbose: print("Calculating Scaff metrics")
            metrics['SNN_testSF'] = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.test_scaffolds_int['SNN'])
            metrics['Frag_testSF'] = FragMetric(**self.kwargs)(gen=mols, pref=self.test_scaffolds_int['Frag'])
            metrics['Scaf_testSF'] = ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.test_scaffolds_int['Scaf'])

        # Target metrics
        if self.target_int is not None:
            if verbose: print("Calculating Target metrics")
            metrics['Novelty_target'] = novelty(gen, self.target, self.pool)
            metrics['AnalogueSimilarity_target'], metrics['AnalogueCoverage_target'] = \
                FingerprintAnaloguesMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.target_int['Analogue'])
            metrics['FG_target'] = FGMetric(**self.kwargs)(pgen={'fgs': fgs}, pref=self.target_int['FG'])
            metrics['RS_target'] = RSMetric(**self.kwargs)(pgen={'rss': rss}, pref=self.target_int['RS'])
            metrics['SNN_target'] = SNNMetric(**self.kwargs)(pgen={'fps': mol_fps}, pref=self.target_int['SNN'])
            metrics['Frag_target'] = FragMetric(**self.kwargs)(gen=mols, pref=self.target_int['Frag'])
            metrics['Scaf_target'] = ScafMetric(**self.kwargs)(pgen={'scaf': scaffs}, pref=self.target_int['Scaf'])
            for name, func in [('logP', logP),
                               ('NP', NP),
                               ('SA', SA),
                               ('QED', QED),
                               ('weight', weight)]:
                metrics[f'{name}_target'] = WassersteinMetric(func, **self.kwargs)(gen=mols, pref=self.target_int[name])

        return metrics

    def property_distributions(self, gen):
        metrics = {}
        if self.test_int is not None:
            for name in ['logP', 'NP', 'SA', 'QED', 'weight']:
                metrics[f'{name}_test'] = self.test_int[name]['values']
        if self.target_int is not None:
            for name in ['logP', 'NP', 'SA', 'QED', 'weight']:
                metrics[f'{name}_test'] = self.target_int[name]['values']

        gen = remove_invalid(gen, canonize=True, n_jobs=self.n_jobs)
        gen = list(set(gen))
        mols = mapper(self.pool)(get_mol, gen)
        for name, func in [('logP', logP),
                           ('NP', NP),
                           ('SA', SA),
                           ('QED', QED),
                           ('weight', weight)]:

            metrics[name] = WassersteinMetric(func, **self.kwargs).precalc(mols)['values']
        return metrics

    def close_pool(self):
        enable_rdkit_log()
        if self.close_pool:
            self.pool.close()
            self.pool.join()
        return


def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    #kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    #statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
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
                       ('weight', weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics


def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan', p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    assert isinstance(gen[0], rdkit.Chem.rdchem.Mol) or isinstance(gen[0], np.ndarray)

    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    else:
        gen_fps = gen

    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


def se_diversity(gen, k=None, n_jobs=1, fp_type='morgan',
                 dist_threshold=0.65, normalize=True):
    """
    Computes Sphere exclusion diversity i.e. fraction of diverse compounds according to a pre-defined
     Tanimoto distance.

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
        gen = gen[:k]

    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    else:
        gen_fps = gen

    bvs = numpy_fps_to_bitvectors(gen_fps, n_jobs=n_jobs)
    no_diverse = sphere_exclusion(fps=bvs, dist_thresh=dist_threshold)
    if normalize:
        return no_diverse / len(gen)
    else:
        return no_diverse


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


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def novelty(gen, train, n_jobs=1):
    if isinstance(gen[0], rdkit.Chem.rdchem.Mol):
        gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    else:
        gen_smiles = gen
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

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
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)


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

    def metric(self, pref, pgen):
        return analogues_tanimoto(pref['fps'], pgen['fps'],
                                  device=self.device)  # Tuple returned (Frac analogues, analogue coverage)


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
