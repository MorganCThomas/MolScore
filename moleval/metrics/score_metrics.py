import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from molbloom import buy

from moleval.utils import Fingerprints, maxmin_picker
from moleval.metrics.metrics import se_diversity, FingerprintAnaloguesMetric
from moleval.metrics.metrics_utils import compute_scaffold, mapper, neutralize_atoms, get_mol, canonic_smiles, QualityFilter
from moleval.metrics.chemistry_filters import ChemistryFilter

class ScoreMetrics:
    def __init__(self, scores: pd.DataFrame = None, target_smiles: list = None, budget: int = None, valid=True, unique=True , n_jobs=1, benchmark=None):
        self.n_jobs = n_jobs
        self.total = len(scores)
        self.budget = budget if budget else self.total
        self.scores = self._preprocess_scores(scores.copy(deep=True), valid=valid, unique=unique, budget=budget)
        self.benchmark = benchmark
        self.target_smiles = target_smiles if target_smiles else []
        self._preprocess_target()
        self.chemistry_filter = ChemistryFilter(n_jobs=self.n_jobs)
        self._bcf_scores = None
        self._tcf_scores = None
        self._btcf_scores = None
        self._rascorer = None

    @property
    def bcf_scores(self):
        if self._bcf_scores is None:
            self._bcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=True, target=False)]
        return self._bcf_scores

    @property
    def tcf_scores(self):
        if self._tcf_scores is None:
            self._tcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=False, target=True)]
        return self._tcf_scores

    @property
    def btcf_scores(self):
        if self._btcf_scores is None:
            self._btcf_scores = self.scores.iloc[self.chemistry_filter.filter_molecules(self.scores.smiles.tolist(), basic=True, target=True)]
        return self._btcf_scores

    @property
    def RAscorer(self):
        if self._rascorer is None:
            from molscore.scoring_functions import RAScore_XGB
            self._rascorer = RAScore_XGB()
        return self._rascorer

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

    def tyield(self, endpoint, threshold, scaffold=False, basic_filter=False, target_filter=False):
        """Threshold yield"""
        # Filter by chemistry
        tdf = self.filter(basic=basic_filter, target=target_filter)
        # Get number of hits
        hits = tdf.loc[tdf[endpoint] >= threshold]
        if scaffold:
            hits = hits.scaffold.dropna().unique()
        return len(hits) / self.budget

    def tyield_auc(self, endpoint, threshold, window=100, extrapolate=True, scaffold=False, basic_filter=False, target_filter=False, return_trajectory=False):
        """Return the AUC of the thresholded yield"""
        # Filter by chemistry
        tdf = self.filter(basic=basic_filter, target=target_filter)

        cumsum = 0
        prev = 0
        called = 0
        indices = []
        yields = []
        # Per log freq
        for idx in range(window, min(len(tdf), self.budget), window):
            temp_result = tdf.iloc[:idx]
            # Get number of hits
            temp_hits = temp_result.loc[temp_result[endpoint] >= threshold]
            if scaffold:
                temp_hits = temp_hits.scaffold.dropna().unique()
            temp_yield = len(temp_hits)
            cumsum += window * (temp_yield + prev) / 2
            prev = temp_yield
            called = idx
            indices.append(idx)
            yields.append(temp_yield)
        # Final cumsum
        hits = tdf.loc[tdf[endpoint] >= threshold]
        if scaffold:
            hits = hits.scaffold.dropna().unique()
        tyield = len(hits)
        cumsum += (len(tdf) - called) * (tyield + prev) / 2
        indices.append(len(tdf))
        yields.append(tyield)
        # If finished early, extrapolate
        if extrapolate and len(tdf) < self.budget:
            cumsum += (self.budget - len(tdf)) * tyield
            indices.append(self.budget)
            yields.append(tyield)
        if return_trajectory:
            return cumsum / self.budget, indices, yields
        else:
            return cumsum / self.budget

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

    def guacamol_score(self, endpoint):
        task = self.scores.task.unique()[0]
        if any([task.lower().startswith(name) for name in [
            "aripiprazole",
            "albuterol",
            "mestranol",
            "median",
            "osimertinib",
            "fexofenadine",
            "ranolazine",
            "perindopril",
            "amlodipine",
            "sitagliptin",
            "zaleplon",
            "valsartan",
            "deco",
            "scaffold",
            "factor_xa_like_scaffold",
            "gcca1_like_scaffold",
            "lorlati_like_scaffold",
            "pde5_scaffold"]]):
            top1, top10, top100 = self.top_avg(top_n=[1, 10, 100], endpoint=endpoint, basic_filter=False, target_filter=False)
            score = np.mean([top1, top10, top100])
        elif any([task.lower().startswith(name) for name in [
            "celecoxib",
            "troglitazone",
            "thiothixene"]]):
            score, = self.top_avg(top_n=[1], endpoint=endpoint, basic_filter=False, target_filter=False)
        elif task == "C11H24":
            score, = self.top_avg(top_n=[159], endpoint=endpoint, basic_filter=False, target_filter=False)
        elif task == "C9H10N2O2PF2Cl":
            score, = self.top_avg(top_n=[250], endpoint=endpoint, basic_filter=False, target_filter=False)
        else:
            print(f"Unknown GuacaMol task {task}, returning uniform specification")
            top1, top10, top100 = self.top_avg(top_n=[1, 10, 100], endpoint=endpoint, basic_filter=False, target_filter=False)
            score = np.mean([top1, top10, top100])
        return score

    def add_benchmark_metrics(self, endpoint):
        benchmark_metrics = {}
        if self.benchmark == "MolOpt":
            # Right now all Molopt metrics are already computed
            pass
        elif self.benchmark == "GuacaMol":
            # Score
            benchmark_metrics["GuacaMol_Score"] = self.guacamol_score(endpoint=endpoint)
            # Quality
            qf = QualityFilter(n_jobs=self.n_jobs)
            top100_mols = self.scores.sort_values(by=endpoint, ascending=False)['smiles'].iloc[:100]
            if len(top100_mols) < 100:
                print(f"Less than 100 molecules to score for GuacaMol Quality, returning 0")
                benchmark_metrics["GuacaMol_Quality"] = 0
            else:
                benchmark_metrics["GuacaMol_Quality"] = qf.score_mols(top100_mols)
        elif self.benchmark == "GuacaMol_Scaffold":
            # Score
            benchmark_metrics["GuacaMol_Score"] = self.guacamol_score(endpoint=endpoint)
            # Quality
            qf = QualityFilter(n_jobs=self.n_jobs)
            top100_mols = self.scores.sort_values(by=endpoint, ascending=False)['smiles'].iloc[:100]
            if len(top100_mols) < 100:
                print(f"Less than 100 molecules to score for GuacaMol Quality, returning 0")
                benchmark_metrics["GuacaMol_Quality"] = 0
            else:
                benchmark_metrics["GuacaMol_Quality"] = qf.score_mols(top100_mols)
        #elif self.benchmark == "LibINVENT_Exp3":
        #    # TODO
        #    pass
        else:
            print(f"Benchmark specific metrics for {self.benchmark} have not been defined yet. Nothing further to add.")
        return benchmark_metrics

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
                        prefix+f"Yield {endpoint}": self.tyield(endpoint=endpoint, threshold=thresholds[i], basic_filter=basic, target_filter=target),
                        prefix+f"Yield AUC {endpoint}": self.tyield_auc(endpoint=endpoint, threshold=thresholds[i], window=100, extrapolate=True, basic_filter=basic, target_filter=target),
                        prefix+f"Yield Scaffold {endpoint}": self.tyield(endpoint=endpoint, threshold=thresholds[i], scaffold=True, basic_filter=basic, target_filter=target),
                        prefix+f"Yield AUC Scaffold {endpoint}": self.tyield_auc(endpoint=endpoint, threshold=thresholds[i], scaffold=True, window=100, extrapolate=True, basic_filter=basic, target_filter=target)
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
            # ----- Property related
            if len(gen_smiles) >= 1000:
                metrics[prefix+"Diversity (SEDiv@1k)"] = se_diversity(gen_smiles, k=1000, n_jobs=self.n_jobs)
        # ----- Further property related
        # MCF filters
        metrics['B-CF'] = len(self.bcf_scores) / self.budget
        # Predicted synthesizability (RAScore > 0.5?)
        if run_synthesizability:
            metrics['Predicted Synthesizability'] = np.mean([r['RAScore_pred_proba'] > 0.5 for r in self.RAscorer(self.scores.smiles.to_list())])
        # Purchasability (MolBloom)
        if run_purchasability:
            metrics['Predicted Purchasability'] = np.mean(mapper(self.n_jobs)(buy, self.scores.smiles.tolist()))
        # ----- Add any benchmark related metrics
        if self.benchmark:
            metrics.update(self.add_benchmark_metrics(endpoint=endpoint))
        return metrics

    def plot_endpoint(self, endpoint, x='index', label=None, chemistry_filters_basic=False, chemistry_filter_target=False,
                      window=100):
        """
        Return the axis of a plot
        :param x: Either "index" or "step"
        :param y: Endpoint to plot
        :param label: Label for the legend
        :param chemistry_filters_basic:
        :param chemistry_filters_target:
        :param window: If plotting index, what window to average values over
        """
        if label is None:
            label = endpoint

        tdf = self.filter(basic=chemistry_filters_basic, target=chemistry_filter_target)
        
        if endpoint not in tdf.columns:
            print(f"Couldn't find endpoint {endpoint} for plotting")
            return
        
        if x == 'index':
            tdf['window'] = (np.arange(len(tdf))//window) + 1
            ax = sns.lineplot(data=tdf, x='window', y=endpoint, label=label.capitalize(), palette='husl')
            xlabels = [f'{int(x)*window/1000:.1f}k' for x in ax.get_xticks()]
            ax.set_xticklabels(xlabels)
        
        else:
            ax = sns.lineplot(data=tdf, x=x, y=endpoint, label=label.capitalize(), palette='husl')
        
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel("Value")
        ax.set_xlim(0, None)
        return ax

    def plot_yield(self,
        endpoint, threshold, label=None, window=100, extrapolate=True, scaffold=False,
        chemistry_filters_basic=False, chemistry_filter_target=False
    ):
        """
        Plot the yield according to some endpoint and threshold, example, ratio of molecules with reward > 0.8.
        """
        if label is None:
            label = endpoint

        if endpoint not in self.scores.columns:
            print(f"Couldn't find endpoint {y} for plotting")
            return
            
        tyield, x, y = self.tyield_auc(
            endpoint=endpoint, threshold=threshold, window=window, extrapolate=extrapolate, scaffold=scaffold,
            basic_filter=chemistry_filters_basic, target_filter=chemistry_filter_target, return_trajectory=True
        )
        
        ax = sns.lineplot(x=x,y=y, palette='husl', label=label.capitalize())
        ax.set_ylabel("Yield")
        ax.set_xlabel("Index")
        ax.set_xticklabels([f'{int(x)/1000}k' for x in ax.get_xticks()])
        return ax

    def _get_chemistry(self, tdf, n=5, scaffold=False, selection=None):
        mol_key = 'smiles' if not scaffold else 'scaffold'
        # If diverse or random
        if selection:
            if selection == 'diverse':
                mols = maxmin_picker(dataset=tdf[mol_key].tolist(), n=n)
                mols = [canonic_smiles(m) for m in mols]
            elif selection == 'range':
                idxs = np.linspace(0, len(tdf)-1, n).astype(int)
                mols = tdf[mol_key].iloc[idxs].tolist()
            elif selection == 'random':
                mols = tdf[mol_key].sample(n).tolist()
            elif selection == 'similar':
                assert self.target_smiles, "Target SMILES are needed to identify similar chemistry"
                # Calculate fps
                ECFP = partial(Fingerprints.ECFP4, nBits=1024)
                query_fps = [fp for fp in mapper(self.n_jobs)(ECFP, tdf[mol_key].tolist())]
                ref_fps = [fp for fp in mapper(self.n_jobs)(ECFP, self.target_smiles if not scaffold else self.target_scaffolds)]
                # Calculate SNNs
                tdf['SNNs'] = [DataStructs.BulkTanimotoSimilarity(qfp, ref_fps) for qfp in query_fps]
                # Sort by SNN
                mols = tdf.sort_values(by='SNNs', ascending=False)[mol_key].iloc[:n].tolist()
            else:
                raise ValueError(f"Unknown type of selection method {selection}, please select out of [diverse, random, similar]")
        else: # Top
            mols = tdf[mol_key].iloc[:n].tolist()
        return mols

    def get_chemistry(
        self,
        endpoint,
        n=5,
        scaffold=False,
        window=100,
        selection=None,
        chemistry_filters_basic=False,
        chemistry_filters_target=False,
        bad_only=False
        ):
        """
        Return some example chemistry
        :param endpoint: Used to sort molecules in descending order assuming higher is better
        :param n: How many molecules to return
        :param scaffold: Scaffold chemistry
        :param window: A subset to search in, e.g., 100 results in a random selection within the top 100 molecules.
        :param selection: How to select molecules in the window. Random, Diverse, Similarity to target or None (Top N)
        :param chemistry_filters_basic: 
        :param chemistry_filters_target:
        :param bad_only: Only return chemistry that fails chemistry_filters_basic
        """
        tdf = self.scores
        # Chemisry Filters
        if bad_only:
            tdf = tdf.loc[~tdf.index.isin(self.filter(basic=True, target=False).index)]
        else:
            tdf = self.filter(basic=chemistry_filters_basic, target=chemistry_filters_target)
        
        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)
        
        # If scaffold, drop duplicate scaffolds (largest score is kept)
        if scaffold:
            tdf = tdf.drop_duplicates(subset='scaffold')
        
        # Subset window
        if window:
            tdf = tdf.iloc[:window]
            
        return self._get_chemistry(tdf, n=n, scaffold=scaffold, selection=selection)

    def get_chemistry_trajectory(
        self,
        endpoint,
        n=5,
        scaffold=False,
        window=100,
        selection=None,
        chemistry_filters_basic=False,
        chemistry_filters_target=False,
        bad_only=False
        ):
        """
        The same as get_chemistry but repeated over the course of de novo generation
        """
        tdf = self.scores
        # Chemisry Filters
        if bad_only:
            import pdb; pdb.set_trace()
            bad_idxs = [i for i in range(len(tdf)) if i not in self.chemistry_filter.filter_molecules(tdf.smiles.tolist(), basic=True, target=False)]
            tdf = tdf.iloc[bad_idxs]
        else:
            tdf = self.filter(basic=chemistry_filters_basic, target=chemistry_filters_target)
        
        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)
        
        # If scaffold, drop duplicate scaffolds (largest score is kept)
        if scaffold:
            tdf = tdf.drop_duplicates(subset='scaffold')

        trajectory_mols = []
        for idx in range(0, len(tdf), window):
            trajectory_mols.append(self._get_chemistry(tdf.iloc[idx:idx+window], n=n, scaffold=scaffold, selection=selection))
        return trajectory_mols

        