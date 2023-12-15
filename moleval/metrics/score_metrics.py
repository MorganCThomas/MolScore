import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from molbloom import buy

from moleval.metrics.metrics import se_diversity, FingerprintAnaloguesMetric
from moleval.metrics.metrics_utils import compute_scaffold, mapper, neutralize_atoms
from moleval.metrics.chemistry_filters import ChemistryFilter

class ScoreMetrics:
    def __init__(self, scores: pd.DataFrame = None, target_smiles: list = None, budget: int = None, valid=True, unique=True , n_jobs=1):
        self.n_jobs = n_jobs
        self.total = len(scores)
        self.budget = budget if budget else self.total
        self.scores = self._preprocess_scores(scores.copy(deep=True), valid=valid, unique=unique, budget=budget)
        self.target_smiles = target_smiles if target_smiles else []
        self._preprocess_target()
        self.chemistry_filter = ChemistryFilter()
        self._bcf_scores = None
        self._tcf_scores = None
        self._btcf_scores = None

    @property
    def bcf_scores(self):
        if self._bcf_scores is None:
            self._bcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=True, target=False)]
        return self._bcf_scores

    @property
    def tcf_scores(self):
        if self._tcf_scores is None:
            self._tcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=False, target=True)]
        return self.tcf_scores

    @property
    def btcf_scores(self):
        if self._btcf_scores is None:
            self._btcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=True, target=True)]
        return self._btcf_scores

    def filter(self, basic=True, target=False):
        if basic and not target:
            return self.bcf_scores
        elif target and not basic:
            return self.tcf_scores
        elif basic and target:
            return self.btcf_scores
        else:
            return self.scores

    def _preprocess_scores(self, scores, valid=True, unique=True, budget=None):
        if budget:
            scores = scores.iloc[:budget]
        if valid:
            scores.loc[:, 'valid'] = scores['valid'].astype(bool)
            scores = scores.loc[scores.valid == True]
        if unique:
            scores.loc[:, 'unique'] = scores['unique'].astype(bool)
            scores = scores.loc[scores.unique == True]
        # Add scaffold column if not present
        if "scaffold" not in scores.columns:
            get_scaff = partial(compute_scaffold, min_rings=1)
            scaffs = mapper(self.n_jobs)(get_scaff, scores.smiles.tolist())
            scores["scaffold"] = scaffs
        return scores

    def _preprocess_target(self):
        self.target_smiles = [smi for smi in mapper(self.n_jobs)(neutralize_atoms, self.target_smiles) if smi is not None]
        get_scaff = partial(compute_scaffold, min_rings=1)
        self.target_scaffolds = mapper(self.n_jobs)(get_scaff, self.target_smiles)
        # TODO fps?

    def top_avg(self, top_n=[1, 10, 100], endpoint=None, basic_filter=False, target_filter=False):
        """Return the average score of the top n molecules"""
        # Filter by chemistry
        tdf = self.filter(basic=basic_filter, target=target_filter)
        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)
        # Get top n
        results = []
        for n in top_n:
            results.append(tdf.iloc[:n][endpoint].mean())
        return results

    def top_auc(self, top_n=[1, 10, 100], endpoint=None, window=100, extrapolate=True, basic_filter=False, target_filter=False):
        """Return the area under the curve of the top n molecules"""
        # Filter by chemistry
        tdf = self.filter(basic=basic_filter, target=target_filter)

        cumsum = [0]*len(top_n)
        prev = [0]*len(top_n)
        called = [0]*len(top_n)
        # Per log freq
        for idx in range(window, min(len(tdf), self.budget), window):
            temp_result = tdf.iloc[:idx]
            # Order by endpoint
            temp_result = temp_result.sort_values(by=endpoint, ascending=False)
            for i, n in enumerate(top_n):
                n_now = temp_result.iloc[:n][endpoint].mean()
                cumsum[i] += window * (n_now + prev[i]) / 2
                prev[i] = n_now
                called[i] = idx
        # Final cumsum
        temp_result = tdf.sort_values(by=endpoint, ascending=False)
        for i, n in enumerate(top_n):
            n_now = temp_result.iloc[:n][endpoint].mean()
            temp_result = temp_result[:n]
            # Compute AUC
            cumsum[i] += (len(tdf) - called[i]) * (n_now + prev[i]) / 2
            # If finished early, extrapolate
            if extrapolate and len(tdf) < self.budget:
                cumsum[i] += (self.budget - len(tdf)) * n_now
        return [x/self.budget for x in cumsum]

    def mol_yield(self, endpoint, threshold, scaffold=False, basic_filter=False, target_filter=False):
        # Filter by chemistry
        tdf = self.filter(basic=basic_filter, target=target_filter)
        # Get number of hits
        hits = tdf.loc[tdf[endpoint] >= threshold]
        if scaffold:
            hits = hits.scaffold.dropna().unique()
        return len(hits) / self.budget

    def targets_rediscovered(self, smiles, scaffold=False):
        # Neutralize & canonize smiles
        smiles = mapper(self.n_jobs)(neutralize_atoms, smiles)
        if scaffold:
            target = self.target_scaffolds
        else:
            target = self.target_smiles
        # Put them both in sets
        target = set(target)
        smiles = set(smiles)
        # Compute intersection
        return len(target.intersection(smiles))

    def get_metrics(self, endpoints=[], thresholds=[], chemistry_filters_basic=False, chemistry_filter_target=False, run_synthesizability=False, run_purchasability=False):
        # NOTE endpoints should be normalized between 0 (bad) and 1 (good) in a standardized, comparable way
        metrics =  {}

        filters = [[False, False]]
        if chemistry_filters_basic: filters.append([True, False])
        if chemistry_filter_target: filters.append([False, True])
        if chemistry_filters_basic and chemistry_filter_target: filters.append([True, True])

        for basic, target in filters:
            prefix = ""
            if basic:
                prefix = "B-CF "
            if target and not basic:
                prefix = "T-CF "
            if basic and target:
                prefix = "B&T-CF "
            # ----- Endpoint related
            for i, endpoint in enumerate(endpoints):
                # Top avg score
                top1, top10, top100 = self.top_avg(top_n=[1, 10, 100], endpoint=endpoint, basic_filter=basic, target_filter=target)
                metrics.update({
                    prefix+f"Top-1 Avg {endpoint}": top1,
                    prefix+f"Top-10 Avg {endpoint}": top10,
                    prefix+f"Top-100 Avg {endpoint}": top100
                    })
                # Top AUC
                top1, top10, top100 = self.top_auc(top_n=[1, 10, 100], endpoint=endpoint, window=100, extrapolate=True, basic_filter=basic, target_filter=target)
                metrics.update({
                    prefix+f"Top-1 AUC {endpoint}": top1,
                    prefix+f"Top-10 AUC {endpoint}": top10,
                    prefix+f"Top-100 AUC {endpoint}": top100
                    })
                # Yield ('Hits' / 'Budget')
                try:
                    metrics.update({
                        prefix+f"Yield {endpoint}": self.mol_yield(endpoint=endpoint, threshold=thresholds[i], basic_filter=basic, target_filter=target),
                        prefix+f"Yield Scaffold {endpoint}": self.mol_yield(endpoint=endpoint, threshold=thresholds[i], scaffold=True, basic_filter=basic, target_filter=target)
                        })
                except IndexError:
                    pass
            # ----- Target related
            gen_smiles = self.filter(basic=basic, target=target).smiles.tolist()
            if self.target_smiles:
                # Rediscovery rate and ratio
                metrics.update({
                    prefix+"Rediscovery Rate": self.targets_rediscovered(gen_smiles, scaffold=False) / self.budget,
                    prefix+"Rediscovery Rate Scaffold": self.targets_rediscovered(gen_smiles, scaffold=True) / self.budget,
                    prefix+"Rediscovered Ratio": self.targets_rediscovered(gen_smiles, scaffold=False) / len(self.target_smiles),
                    prefix+"Rediscovered Ratio Scaffold": self.targets_rediscovered(gen_smiles, scaffold=True) / len(self.target_scaffolds)
                })
                # Fingerprint analogues
                gen_ans, ref_ans = FingerprintAnaloguesMetric(n_jobs=self.n_jobs)(gen=gen_smiles, ref=self.target_smiles)
                metrics.update({
                    prefix+"Analogue Rate": gen_ans,
                    prefix+"Analogue Ratio": ref_ans,
                })
                # TODO Sillyness?
            # ----- Property related
            if len(gen_smiles) >= 1000:
                metrics[prefix+"Diversity (SEDiv@1k)"] = se_diversity(gen_smiles, k=1000, n_jobs=self.n_jobs)
        # ----- Further property related
        # MCF filters
        metrics['B-CF'] = len(self.bcf_scores) / self.budget
        # TODO Predicted synthesizability (RAScore > 0.5?)
        if run_synthesizability:
            raise NotImplementedError
        # Purchasability (MolBloom)
        if run_purchasability:
            metrics['Predicted Purchasability'] = np.mean(mapper(self.n_jobs)(buy, self.scores.smiles.tolist()))
        return metrics

    def plot_endpoint(self, x=None, y=None, label=None):
        """Return the axis of a plot"""
        # X = ["Step", "Index"]
        # Y = ["Any endpoint"]
        # TODO endpoint optimization
        raise NotImplementedError

    def plot_yield():
        # TODO
        raise NotImplementedError

    def get_chemistry():
        # TODO top? molecules for each chemistry filter? or a diverse selection? or a random selection? or most similar to target?
        raise NotImplementedError

        