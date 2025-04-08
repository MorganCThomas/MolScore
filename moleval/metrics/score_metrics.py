import re
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
try:
    from molbloom import buy
    _has_molbloom = True
except (ImportError, TypeError) as e:
    print(f"Molbloom incompatible, skipping purchasability score: {e}")
    _has_molbloom = False
    
from rdkit import DataStructs

from moleval.metrics.chemistry_filters import ChemistryBuffer, ChemistryFilter
from moleval.metrics.metrics import FingerprintAnaloguesMetric, se_diversity
from moleval.metrics.metrics_utils import (
    QualityFilter,
    canonic_smiles,
    compute_scaffold,
    mapper,
    neutralize_atoms,
)
from moleval.utils import (
    Fingerprints,
    check_env_jobs,
    get_multiprocessing_context,
    maxmin_picker,
)


class ScoreMetrics:
    
    metric_descriptions = {
        "Valid": "The ratio of valid molecules to the total number of molecules",
        "Unique": "The ratio of unique molecules to the total number of molecules",
        "Top-{N} Avg": "The average score of the top N molecules",
        "Top-{N} Avg (Div)": "The average score of the top N diverse molecules",
        "Top-{N} AUC": "The area under the curve of the top N molecules",
        "Top-{N} AUC (Div)": "The area under the curve of the top N diverse molecules",
        "Yield": "The yield of molecules above a certain threshold",
        "Yield Scaffold": "The yield of scaffolds above a certain threshold",
        "Yield AUC": "The area under the curve of the yield",
        "Yield AUC Scaffold": "The area under the curve of the scaffold yield",
        "Rediscovery Rate": "The number of target molecules rediscovered relative to the budget",
        "Rediscovery Ratio": "The ratio of target molecules rediscovered relative to target molecules",
        "Rediscovery Rate Scaffold": "The number of target scaffolds rediscovered",
        "Rediscovery Ratio Scaffold": "The ratio of target scaffolds rediscovered",
        "Analogue Rate": "The number of target analogues found relative to the budget",
        "Analogue Ratio": "The ratio of target analogues found relative to target molecules",
        "Diversity (SEDiv@1k)": "The diversity of the molecules by sphere exclusion diversity at 1k",
        "Predicted Synthesizability": "The predicted synthesizability of the molecules by RAScore",
        "Predicted Purchasability": "The predicted purchasability of the molecules in ZINC",
        "B-CF": "The ratio of molecules passing basic chemistry filters",
        "T-CF": "The ratio of molecules passing target chemistry filters",
        "B&T-CF": "The ratio of molecules passing basic and target chemistry filters",
    }
    
    def __init__(
        self,
        scores: pd.DataFrame = None,
        reference_smiles: list = [],
        budget: int = None,
        valid=True,
        unique=True,
        n_jobs=1,
        benchmark=None,
    ):
        """This class facilitates the calculation of metrics (and benchmark metrics) 
        from the scores dataframe returned from MolScore. Additionally contains functions
        to plot metrics and select chemistry examples.
        :param scores: pd.DataFrame, the scores dataframe returned from MolScore
        :param reference_smiles: list, a list of reference SMILES to compare against as 'Target' property chemistry
        :param budget: int, the budget of the experiment
        :param valid: bool, whether to truncate to only valid smiles
        :param unique: bool, whether to truncate to only unique smiles
        :param n_jobs: int, number of jobs to run in parallel
        :param benchmark: str, a benchmark to calculate specific metrics for
        """
        
        # Pre-process scores
        self.n_jobs = check_env_jobs(n_jobs)
        self.total = len(scores)
        self.budget = budget if budget else self.total
        self.scores = self._preprocess_scores(
            scores.copy(deep=True), valid=valid, unique=unique, budget=budget
        )
        self.benchmark = benchmark
        
        # Pre-process reference smiles
        self.has_reference_smiles = False
        self.reference_smiles = reference_smiles
        if reference_smiles:
            self.reference_smiles, self.reference_scaffolds = self._preprocess_smiles(
                self.reference_smiles
            )
            self.has_reference_smiles = True
        
        # Setup chemistry filter
        self.chemistry_filter = ChemistryFilter(
            target=self.reference_smiles, n_jobs=self.n_jobs
        )
        
        # Parameters
        self._bcf_scores = None
        self._tcf_scores = None
        self._btcf_scores = None
        self._rascorer = None

    def _reinitialize(
        self, scores, budget: int = None, benchmark: str = None, n_jobs: int = None
    ):
        # Re-initialize without recomputing reference smiles
        self.total = len(scores)
        if budget:
            self.budget = budget
        if not self.budget:
            self.total
        if benchmark:
            self.benchmark = benchmark
        if n_jobs:
            self.n_jobs = n_jobs
            self.chemistry_filter = ChemistryFilter(
                target=self.reference_smiles, n_jobs=self.n_jobs
            )
        self.scores = self._preprocess_scores(
            scores.copy(deep=True), valid=True, unique=True, budget=self.budget
        )
        self._bcf_scores = None
        self._tcf_scores = None
        self._btcf_scores = None

    @property
    def bcf_scores(self):
        if self._bcf_scores is None:
            self._bcf_scores = self.scores.iloc[
                self.chemistry_filter.filter_molecules(
                    self.scores.smiles.tolist(), basic=True, target=False
                )
            ]
        return self._bcf_scores

    @property
    def tcf_scores(self):
        if self._tcf_scores is None:
            self._tcf_scores = self.scores.iloc[
                self.chemistry_filter.filter_molecules(
                    self.scores.smiles.tolist(), basic=False, target=True
                )
            ]
        return self._tcf_scores

    @property
    def btcf_scores(self):
        if self._btcf_scores is None:
            self._btcf_scores = self.scores.iloc[
                self.chemistry_filter.filter_molecules(
                    self.scores.smiles.tolist(), basic=True, target=True
                )
            ]
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
        # Truncate scores df to budget
        if budget:
            scores = scores.iloc[:budget]
        len_all = len(scores)
        
        # Truncate to valid only molecules and calculate valid ratio
        if isinstance(scores.valid.dtype, np.dtypes.ObjectDType):
            valid_value = "true" # Back compatability
        elif isinstance(scores.valid.dtype, np.dtypes.BoolDType):
            valid_value = True
        else:
            raise ValueError("Valid column has un unrecognised dtype")
        if valid:
            scores = scores.loc[scores.valid == valid_value]
            self.valid_ratio = len(scores) / len_all
        else:
            self.valid_ratio = (scores.valid == valid_value).sum() / len_all
        len_valid = len(scores)
            
        # Truncate to unique only molecules and calculate unique ratio
        if isinstance(scores.unique.dtype, np.dtypes.ObjectDType):
            unique_value = "true" # Back compatability
        elif isinstance(scores.unique.dtype, np.dtypes.BoolDType):
            unique_value = True
        else:
            raise ValueError("Unique column has un unrecognised dtype")
        if unique:
            scores = scores.loc[scores.unique == unique_value]
            self.unique_ratio = len(scores) / len_valid
        else:
            self.unique_ratio = (scores.unique == unique_value).sum() / len(scores)
            
        # Add scaffold column if not present
        if "smiles" in scores.columns:
            self.has_smiles = True
            if "scaffold" in scores.columns:
                self.has_scaffold = True
            else:
                self.has_scaffold = False
        else:
            self.has_smiles = False
        if self.has_smiles and not self.has_scaffold:
            get_scaff = partial(compute_scaffold, min_rings=1)
            scaffs = mapper(self.n_jobs)(get_scaff, scores.smiles.tolist())
            scores["scaffold"] = scaffs
            self.has_scaffold = True
        
        return scores

    def _preprocess_smiles(self, smiles):
        smiles = [
            smi
            for smi in mapper(self.n_jobs)(neutralize_atoms, smiles)
            if smi is not None
        ]
        
        get_scaff = partial(compute_scaffold, min_rings=1)
        scaffolds = mapper(self.n_jobs)(get_scaff, smiles)
        return smiles, scaffolds

    @staticmethod
    def top_avg(
        scores,
        top_n=[1, 10, 100],
        endpoint=None,
        diverse=False,
        prefix="",
        queue=None,
    ):
        """Return the average score of the top n molecules"""
        # Filter by chemistry
        tdf = scores
        buffer = ChemistryBuffer(buffer_size=max(top_n))
        if diverse:
            buffer.update_from_score_metrics(df=tdf, endpoint=endpoint)
        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)
        # Get top n
        results = []
        for n in top_n:
            if diverse:
                results.append(buffer.top_n(n))
            else:
                results.append(tdf.iloc[:n][endpoint].mean())

        # Return as dictionary
        if diverse:
            output = [
                (prefix + f"Top-{n} Avg (Div) {endpoint}", r)
                for n, r in zip(top_n, results)
            ]
        else:
            output = [
                (prefix + f"Top-{n} Avg {endpoint}", r)
                for n, r in zip(top_n, results)
            ]
        if queue:
            for o in output:
                queue.put(o)
        else:
            return dict(output)

    @staticmethod
    def top_auc(
        scores,
        budget,
        top_n=[1, 10, 100],
        endpoint=None,
        window=100,
        extrapolate=True,
        diverse=False,
        return_trajectory=False,
        prefix="",
        queue=None,
    ):
        """Return the area under the curve of the top n molecules"""
        # Filter by chemistry
        tdf = scores
        cumsum = [0] * len(top_n)
        prev = [0] * len(top_n)
        called = 0
        indices = [[0] for _ in range(len(top_n))]
        auc_values = [[0] for _ in range(len(top_n))]
        buffer = ChemistryBuffer(buffer_size=max(top_n))
        # Per log freq
        for idx in range(window, min(tdf.index.max(), budget), window):
            if diverse:
                # Buffer keeps a memory so only need the latest window
                buffer.update_from_score_metrics(
                    df=tdf.loc[idx - window : idx], endpoint=endpoint
                )
                for i, n in enumerate(top_n):
                    n_now = buffer.top_n(n)
                    cumsum[i] += window * ((n_now + prev[i]) / 2)
                    prev[i] = n_now
                    called = idx
                    indices[i].append(window)
                    auc_values[i].append(n_now)
            else:
                # Order by endpoint up till index
                temp_result = tdf.loc[:idx]
                temp_result = temp_result.sort_values(by=endpoint, ascending=False)
                for i, n in enumerate(top_n):
                    n_now = temp_result.iloc[:n][endpoint].mean()
                    cumsum[i] += window * ((n_now + prev[i]) / 2)
                    prev[i] = n_now
                    called = idx
                    indices[i].append(window)
                    auc_values[i].append(n_now)
        # Final cumsum
        if diverse:
            buffer.update_from_score_metrics(
                df=tdf.loc[called : tdf.index.max()], endpoint=endpoint
            )
            for i, n in enumerate(top_n):
                n_now = buffer.top_n(n)
                # Compute AUC
                cumsum[i] += (tdf.index.max() - called) * ((n_now + prev[i]) / 2)
                indices[i].append(tdf.index.max())
                auc_values[i].append(n_now)
                # If finished early, extrapolate
                if extrapolate and (tdf.index.max() < budget):
                    cumsum[i] += (budget - tdf.index.max()) * n_now
                    indices[i].append(budget)
                    auc_values[i].append(n_now)
        else:
            temp_result = tdf.sort_values(by=endpoint, ascending=False)
            for i, n in enumerate(top_n):
                n_now = temp_result.iloc[:n][endpoint].mean()
                # Compute AUC
                cumsum[i] += (tdf.index.max() - called) * ((n_now + prev[i]) / 2)
                indices[i].append(tdf.index.max())
                auc_values[i].append(n_now)
                # If finished early, extrapolate
                if extrapolate and (tdf.index.max() < budget):
                    cumsum[i] += (budget - tdf.index.max()) * n_now
                    indices[i].append(budget)
                    auc_values[i].append(n_now)

        if return_trajectory:
            return [x / budget for x in cumsum], indices, auc_values

        # Return as dictionary
        if diverse:
            output = [
                (prefix + f"Top-{n} AUC (Div) {endpoint}", x / budget)
                for n, x in zip(top_n, cumsum)
            ]
        else:
            output = [
                (prefix + f"Top-{n} AUC {endpoint}", x / budget)
                for n, x in zip(top_n, cumsum)
            ]
        if queue:
            for o in output:
                queue.put(o)
        else:
            return dict(output)

    @staticmethod
    def tyield(
        scores,
        budget,
        endpoint,
        threshold,
        scaffold=False,
        prefix="",
        queue=None,
    ):
        """Threshold yield"""
        # Filter by chemistry
        tdf = scores
        # Get number of hits
        hits = tdf.loc[tdf[endpoint] >= threshold]
        if scaffold:
            hits = hits.scaffold.dropna().unique()

        # Return as dictionary
        if scaffold:
            output = (prefix + f"Yield Scaffold {endpoint}", len(hits) / budget)
        else:
            output = (prefix + f"Yield {endpoint}", len(hits) / budget)
        if queue:
            queue.put(output)
        else:
            return dict([output])

    @staticmethod
    def tyield_auc(
        scores,
        budget,
        endpoint,
        threshold,
        window=100,
        extrapolate=True,
        scaffold=False,
        return_trajectory=False,
        prefix="",
        queue=None,
    ):
        """Return the AUC of the thresholded yield"""
        # Filter by chemistry
        tdf = scores

        cumsum = 0
        prev = 0
        called = 0
        indices = []
        yields = []
        # Per log freq
        for idx in range(window, min(tdf.index.max(), budget), window):
            temp_result = tdf.loc[:idx]
            # Get number of hits
            temp_hits = temp_result.loc[temp_result[endpoint] >= threshold]
            if scaffold:
                temp_hits = temp_hits.scaffold.dropna().unique()
            temp_yield = len(temp_hits) / idx
            cumsum += window * ((temp_yield + prev) / 2)
            prev = temp_yield
            called = idx
            indices.append(idx)
            yields.append(temp_yield)
        # Final cumsum
        hits = tdf.loc[tdf[endpoint] >= threshold]
        if scaffold:
            hits = hits.scaffold.dropna().unique()
        tyield = len(hits) / tdf.index.max()
        cumsum += (tdf.index.max() - called) * (tyield + prev) / 2
        indices.append(tdf.index.max())
        yields.append(tyield)
        # If finished early, extrapolate
        if extrapolate and tdf.index.max() < budget:
            cumsum += (budget - tdf.index.max()) * tyield
            indices.append(budget)
            yields.append(tyield)
        if return_trajectory:
            return cumsum / budget, indices, yields

        # Return as dictionary
        if scaffold:
            output = (prefix + f"Yield AUC Scaffold {endpoint}", cumsum / budget)
        else:
            output = (prefix + f"Yield AUC {endpoint}", cumsum / budget)
        if queue:
            queue.put(output)
        else:
            return dict([output])

    @staticmethod
    def targets_rediscovered(
        smiles, target_smiles, scaffolds=[], target_scaffolds=[]
    ):
        """Rediscovery of provided target molecules and scaffolds"""
        query = set(smiles)
        target = set(target_smiles)
        rediscovered_smiles = len(target.intersection(query))
        rediscovered_scaffolds = None
        if scaffolds and target_scaffolds:
            query = set(scaffolds)
            target = set(target_scaffolds)
            rediscovered_scaffolds = len(target.intersection(query))
        return rediscovered_smiles, rediscovered_scaffolds
    
    def molopt_score(self, endpoint):
        metrics = self.run_metrics(
            endpoints=[endpoint],
            include=[
                "Top-1 Avg",
                "Top-10 Avg",
                "Top-100 Avg",
                "Top-1 AUC",
                "Top-10 AUC",
                "Top-100 AUC",
            ]
        )
        return metrics

    def guacamol_score(self, endpoint):
        # Calculate Score
        task = self.scores.task.unique()[0]
        tdf = self.scores.copy()
        if any(
            [
                task.lower().startswith(name)
                for name in [
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
                    "pde5_scaffold",
                ]
            ]
        ):
            top1, top10, top100 = self.top_avg(
                scores=tdf,
                top_n=[1, 10, 100],
                endpoint=endpoint,
            ).values()
            score = np.mean([top1, top10, top100])
        elif any(
            [
                task.lower().startswith(name)
                for name in ["celecoxib", "troglitazone", "thiothixene"]
            ]
        ):
            (score,) = self.top_avg(
                scores=tdf,
                top_n=[1],
                endpoint=endpoint,
            ).values()
        elif task == "C11H24":
            (score,) = self.top_avg(
                scores=tdf,
                top_n=[159],
                endpoint=endpoint,
            ).values()
        elif task == "C9H10N2O2PF2Cl":
            (score,) = self.top_avg(
                scores=tdf,
                top_n=[250],
                endpoint=endpoint,
            ).values()
        else:
            print(f"Unknown GuacaMol task {task}, returning uniform specification")
            top1, top10, top100 = self.top_avg(
                scores=tdf,
                top_n=[1, 10, 100],
                endpoint=endpoint,
            ).values()
            score = np.mean([top1, top10, top100])
        # Calculate Quality
        qf = QualityFilter(n_jobs=self.n_jobs)
        top100_mols = self.scores.sort_values(by=endpoint, ascending=False)[
            "smiles"
        ].iloc[:100]
        if len(top100_mols) < 100:
            print("Less than 100 molecules to score for GuacaMol Quality, returning 0")
            quality_score = 0
        else:
            quality_score = qf.score_mols(top100_mols)
        metrics = {
            "GuacaMol_Score": score,
            "GuacaMol_Quality": quality_score,
        }
        return metrics

    def libinvent_score(self, endpoint):
        task = self.scores.task.unique()[0]
        N = len(self.scores)
        found = self.scores.loc[(self.scores.DRD2_pred_proba >= 0.4)]
        N_found = len(found)
        avg_score = found[endpoint].mean()
        metrics = {
            "LibINVENT_N": N_found,
            "LibINVENT_Yield": N_found / N,
            "LibINVENT_Avg_Score": avg_score,
        }
        if task.lower() == "drd2_subfilt_df":
            pass
        elif task.lower() == "drd2_selrf_subfilt_df":
            N_reaction_satisfied = len(
                self.scores.loc[self.scores.ReactionFilter_score == 1.0]
            )
            metrics["LibINVENT_Fully_Satisfied"] = N_reaction_satisfied / N
        else:
            print(f"Unknown LibINVENT task {task}")

        return metrics

    def molexp_score(self):
        tdf = self.scores.copy()
        sim_endpoints = [
            c for c in self.scores.columns if re.search("_Cmpd[0-9]*_Sim", c)
        ]
        top_avgs = []
        top_aucs = []
        for endpoint in sim_endpoints:
            # Calculate top AVG
            top1, top10, top100 = self.top_avg(
                scores=tdf,
                top_n=[1, 10, 100],
                endpoint=endpoint,
            ).values()
            top_avgs.append([top1, top10, top100])
            # Calculate top AUC
            top1, top10, top100 = self.top_auc(
                scores=tdf,
                budget=self.budget,
                top_n=[1, 10, 100],
                endpoint=endpoint,
                window=100,
                extrapolate=True,
            ).values()
            top_aucs.append([top1, top10, top100])
        # Aggregate and take product
        top_avgs = np.vstack(top_avgs).prod(0)
        top_aucs = np.vstack(top_aucs).prod(0)
        metrics = {
            "Top-1 Avg (Exp)": top_avgs[0],
            "Top-10 Avg (Exp)": top_avgs[1],
            "Top-100 Avg (Exp)": top_avgs[2],
            "Top-1 AUC (Exp)": top_aucs[0],
            "Top-10 AUC (Exp)": top_aucs[1],
            "Top-100 AUC (Exp)": top_aucs[2],
        }
        return metrics
    
    def run_benchmark_metrics(self, endpoint):
        benchmark_metrics = {}
        if self.benchmark.startswith("MolOpt"):
            benchmark_metrics.update(self.molopt_score(endpoint=endpoint))
        elif self.benchmark.startswith("MolExp"):
            benchmark_metrics.update(self.molexp_score())
        elif self.benchmark in ["GuacaMol", "GuacaMol_Scaffold"]:
            benchmark_metrics.update(self.guacamol_score(endpoint=endpoint))
        elif self.benchmark == "LibINVENT_Exp1":
            benchmark_metrics.update(self.libinvent_score(endpoint=endpoint))
        else:
            print(
                f"Benchmark specific metrics for {self.benchmark} have not been defined yet. Nothing further to add."
            )
        return benchmark_metrics   

    def run_metrics(
        self,
        endpoints=[],
        thresholds=[], 
        target_smiles=[],
        include=["Valid", "Unique"],
        chemistry_filter_basic=False,
        chemistry_filter_target=False,
        extrapolate=True,
    ):
        """Calculate metrics.
        :param endpoints: list, the endpoints in scores to calculate metrics for e.g., "Single". NOTE endpoints should be normalized between 0 (bad) and 1 (good) in a standardized, comparable way
        :param thresholds: list, the thresholds corresponding to endpoints to calculate yield metrics for e.g., 0.8
        :param target_smiles: list, target smiles to compare against
        :param include: list, metrics to calculate besides any specified benchmark metrics, description available in ScoreMetrics.metric_descriptions
        :param chemistry_filter_basic: bool, whether to filter by basic drug like chemistry properties
        :param chemistry_filter_target: bool, whether to filter by target chemistry
        :param extrapolate: bool, whether to extrapolate metrics to the budget if fewer molecules exist due to invalid or non-uniqueness
        """

        # Setup parallelisation for internal async processes
        mp = get_multiprocessing_context()
        queue = mp.Queue()
        process_list = []  # (func, kwargs, block)

        # Setup metrics
        metrics = {}
        
        # Function to check include
        def check_top_N(patt):
            patt = re.compile(patt)
            metric_names = [patt.search(m) for m in include if patt.search(m)]
            metric_names = [m for m in metric_names if f"{m.string} {endpoint}" not in metrics]
            top_ns = [int(m.groups()[0]) for m in metric_names]
            return (bool(metric_names), top_ns)
        
        # Valid / unique
        if "Valid" in include:
            metrics.update({"Valid": self.valid_ratio})
        if "Unique" in include:
            metrics.update({"Unique": self.unique_ratio})
            
        # Setup filters to iterate
        score_filters = [("", [False, False])]
        if chemistry_filter_basic:
            score_filters.append(("B-CF ", [True, False]))
        if chemistry_filter_target and not self.has_reference_smiles:
            print("No reference smiles provided for target chemistry filter")
        if chemistry_filter_target and self.has_reference_smiles:
            score_filters.append(("T-CF ", [False, True]))
        if chemistry_filter_basic and chemistry_filter_target and self.has_reference_smiles:
            score_filters.append(("B&T-CF ", [True, True]))

        # Setup target smiles
        if target_smiles and self.has_smiles:
            target_smiles, target_scaffolds = self._preprocess_smiles(target_smiles)

        for prefix, (basic, target) in score_filters:
            # ----- Filter scores
            filtered_scores = self.filter(basic=basic, target=target).copy() # NOTE is .copy() necessary?

            # ----- Endpoint related
            for i, endpoint in enumerate(endpoints):
                # Top-{N} Avg score
                top_name, top_ns = check_top_N(patt=f"Top-([0-9]+) Avg")
                if top_name:
                    process_list.append(
                        (
                            self.top_avg,
                            (filtered_scores, top_ns, endpoint, False, prefix, queue),
                            False,
                        )
                    )
                
                # Top-{N} Avg (Div)
                top_name, top_ns = check_top_N(patt=f"Top-([0-9]+) Avg (Div)")
                if top_name:
                    process_list.append(
                        (
                            self.top_avg,
                            (
                                filtered_scores,
                                top_ns,
                                endpoint,
                                True,
                                prefix,
                                queue,
                            ),
                            False,
                        )
                    )
                    
                # Top-{N} AUC
                top_name, top_ns = check_top_N(patt=f"Top-([0-9]+) AUC")
                if top_name:
                    process_list.append(
                        (
                            self.top_auc,
                            (
                                filtered_scores,
                                self.budget,
                                top_ns,
                                endpoint,
                                100,
                                extrapolate,
                                False,
                                False,
                                prefix,
                                queue,
                            ),
                            False,
                        )
                    )
                    
                # Top-{N} AUC (Div)
                top_name, top_ns = check_top_N(patt=f"Top-([0-9]+) AUC (Div)")
                if top_name:
                    process_list.append(
                        (
                            self.top_auc,
                            (
                                filtered_scores,
                                self.budget,
                                top_ns,
                                endpoint,
                                100,
                                extrapolate,
                                True,
                                False,
                                prefix,
                                queue,
                            ),
                            False,
                        )
                    )
                    
                # Yield (Check a corresponding threshold has been provided)
                try:
                    threshold = thresholds[i]
                except IndexError:
                    threshold = False
                    if any(m.startswith("Yield") for m in include):
                        print(f"No threshold was given for {endpoint}")
            
                if threshold:
                    patt = f"Yield"
                    if (patt in include) and (f"{patt} {endpoint}" not in metrics):
                        process_list.append(
                            (
                                self.tyield,
                                (
                                    filtered_scores,
                                    self.budget,
                                    endpoint,
                                    threshold,
                                    False, # Scaffold
                                    prefix,
                                    queue,
                                ),
                                False,
                            )
                        )
                    patt = f"Yield Scaffold"
                    if (patt in include) and (f"{patt} {endpoint}" not in metrics) and self.has_scaffold:
                        process_list.append(
                            (
                                self.tyield,
                                (
                                    filtered_scores,
                                    self.budget,
                                    endpoint,
                                    threshold,
                                    True, # Scaffold
                                    prefix,
                                    queue,
                                ),
                                False,
                            )
                        )
                    patt = f"Yield AUC"
                    if (patt in include) and (f"{patt} {endpoint}" not in metrics):
                        process_list.append(
                            (
                                self.tyield_auc,
                                (
                                    filtered_scores,
                                    self.budget,
                                    endpoint,
                                    threshold,
                                    100,
                                    True,
                                    False, # Scaffold
                                    False,
                                    prefix,
                                    queue,
                                ),
                                False,
                            )
                        )
                    patt = f"Yield AUC Scaffold"
                    if (patt in include) and (f"{patt} {endpoint}" not in metrics) and self.has_scaffold:
                        process_list.append(
                            (
                                self.tyield_auc,
                                (
                                    filtered_scores,
                                    self.budget,
                                    endpoint,
                                    threshold,
                                    100,
                                    True,
                                    True, # Scaffold
                                    False,
                                    prefix,
                                    queue,
                                ),
                                False,
                            )
                        )

                # ----- Submit and run endpoint related processes
                if self.n_jobs > 1:
                    metrics.update(self._run_async_processes(process_list, queue))
                else:
                    for func, args, _ in process_list:
                        # Don't pass queue
                        metrics.update(func(*args[:-1]))
                    

            # ----- Target related
            if target_smiles and self.has_smiles:
                if any([(m in include) and (m not in metrics) for m in
                       ["Rediscovery Rate", "Rediscovery Ratio",
                        "Rediscovery Rate Scaffold", "Rediscovery Ratio Scaffold"]]):
                
                    # Setup gen smiles
                    gen_smiles = filtered_scores.smiles.tolist()
                    gen_smiles, gen_scaffolds = self._preprocess_smiles(gen_smiles)
                    
                    # Rediscovery rate and ratio
                    rediscovered_smiles, rediscovered_scaffolds = self.targets_rediscovered(
                        gen_smiles, target_smiles, gen_scaffolds, target_scaffolds
                    )
                    metrics.update(
                        {
                            prefix + "Rediscovery Rate": rediscovered_smiles / self.budget,
                            prefix + "Rediscovered Ratio": rediscovered_smiles
                            / len(target_smiles),
                        }
                    )
                    metrics.update(
                        {
                            prefix + "Rediscovery Rate Scaffold": rediscovered_scaffolds
                            / self.budget,
                            prefix
                            + "Rediscovered Ratio Scaffold": rediscovered_scaffolds
                            / len(target_scaffolds),
                        }
                    )
                    
                # Fingerprint analogues
                if any([m in include for m in ["Analogue Rate", "Analogue Ratio"]]):
                    gen_ans, ref_ans = FingerprintAnaloguesMetric(n_jobs=self.n_jobs)(
                        gen=gen_smiles, ref=target_smiles
                    )
                    metrics.update(
                        {
                            prefix + "Analogue Rate": gen_ans,
                            prefix + "Analogue Ratio": ref_ans,
                        }
                    )

            # ----- Property related
            if ("Diversity (SEDiv@1k)" in include) and ("Diversity (SEDiv@1k)" not in metrics) and (len(gen_smiles) >= 1000):
                metrics[prefix + "Diversity (SEDiv@1k)"] = se_diversity(
                    gen_smiles, k=1000, n_jobs=self.n_jobs
                )

        # ----- Further property related
        # MCF filters
        if any([(m in include) and (m not in metrics) for m in ["B-CF", "T-CF", "B&T-CF"]]):
            metrics["B-CF"] = len(self.filter(basic=True, target=False)) / self.budget
            if chemistry_filter_target and self.has_reference_smiles:
                metrics["T-CF"] = len(self.filter(basic=False, target=True)) / self.budget
                metrics["B&T-CF"] = len(self.filter(basic=True, target=True)) / self.budget

        # Predicted synthesizability (RAScore > 0.5?)
        if ("Predicted Synthesizability" in include) and ("Predicted Synthesizability" not in metrics):
            metrics["Predicted Synthesizability"] = np.mean(
                [
                    r["RAScore_pred_proba"] > 0.5
                    for r in self.RAscorer(self.scores.smiles.to_list())
                ]
            )
        # Purchasability (MolBloom)
        if ("Predicted Purchasability" in include) and ("Predicted Purchasability" not in metrics) and _has_molbloom:
            metrics["Predicted Purchasability"] = np.mean(
                mapper(self.n_jobs)(buy, self.scores.smiles.tolist())
            )

        return metrics
    
    def get_metrics(
        self,
        endpoints=[],
        thresholds=[], 
        target_smiles=[],
        include=["Valid", "Unique"],
        chemistry_filter_basic=False,
        chemistry_filter_target=False,
        extrapolate=True
        ):
        """Calculate metrics.
        :param endpoints: list, the endpoints in scores to calculate metrics for e.g., "Single". NOTE endpoints should be normalized between 0 (bad) and 1 (good) in a standardized, comparable way
        :param thresholds: list, the thresholds corresponding to endpoints to calculate yield metrics for e.g., 0.8
        :param target_smiles: list, target smiles to compare against
        :param include: list, metrics to calculate besides any specified benchmark metrics, description available in ScoreMetrics.metric_descriptions
        :param chemistry_filter_basic: bool, whether to filter by basic drug like chemistry properties
        :param chemistry_filter_target: bool, whether to filter by target chemistry
        :param extrapolate: bool, whether to extrapolate metrics to the budget if fewer molecules exist due to invalid or non-uniqueness
        """
        
        metrics = {}
        
        for endpoint in endpoints:
            # First run any benchmark metrics
            metrics.update(self.run_benchmark_metrics(endpoint=endpoint))
        
        # Then run additional specificed metrics
        metrics.update(
            self.run_metrics(
                endpoints=endpoints,
                thresholds=thresholds,
                target_smiles=target_smiles,
                include=include,
                chemistry_filter_basic=chemistry_filter_basic,
                chemistry_filter_target=chemistry_filter_target,
                extrapolate=extrapolate               
            )
        )
        return metrics 

    def _run_async_processes(self, process_list, queue):
        mp = get_multiprocessing_context()
        processes = []
        done = []
        for pargs in process_list:
            # Submit process
            if len(processes) < self.n_jobs:
                p = mp.Process(target=pargs[0], args=pargs[1])
                p.start()
                processes.append(p)

                # If it is blocking, wait for it to finish
                if pargs[2]:
                    p.join()
                    done.append(processes.pop(-1))

            # If out of workers, wait for one to finish
            while len(processes) >= self.n_jobs:
                # Wait for a process to finish
                for i, p in enumerate(processes):
                    if not p.is_alive():
                        p.join()
                        done.append(processes.pop(i))
                        break

        # Wait for all processes to finish
        for i, p in enumerate(processes):
            p.join()
            done.append(processes.pop(i))
            
        # Fetch metrics
        results = []
        while not queue.empty():
            results.append(queue.get())
        results = dict(results)

        return results

    def plot_endpoint(
        self,
        endpoint,
        x="index",
        label=None,
        chemistry_filters_basic=False,
        chemistry_filter_target=False,
        window=100,
    ):
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

        if x == "index":
            tdf["window"] = (np.arange(len(tdf)) // window) + 1
            ax = sns.lineplot(
                data=tdf,
                x="window",
                y=endpoint,
                label=label.capitalize(),
                palette="husl",
            )
            xlabels = [f"{int(x)*window/1000:.0f}k" for x in ax.get_xticks()]
            ax.set_xticklabels(xlabels)

        else:
            ax = sns.lineplot(data=tdf, x=x, y=endpoint, label=label, palette="husl")

        ax.set_xlabel(x.capitalize())
        ax.set_ylabel("Value")
        ax.set_xlim(0, None)
        return ax

    def plot_auc(
        self,
        endpoint,
        top_n=100,
        label=None,
        window=100,
        extrapolate=True,
        chemistry_filters_basic=False,
        chemistry_filter_target=False,
        diverse=False,
    ):
        """
        Plot the AUC of the top n molecules
        """
        if label is None:
            label = endpoint

        if endpoint not in self.scores.columns:
            print(f"Couldn't find endpoint {endpoint} for plotting")
            return

        _, x, y = self.top_auc(
            top_n=[top_n],
            endpoint=endpoint,
            window=window,
            extrapolate=extrapolate,
            basic_filter=chemistry_filters_basic,
            target_filter=chemistry_filter_target,
            diverse=diverse,
            return_trajectory=True,
        )

        ax = sns.lineplot(x=x[0], y=y[0], palette="husl", label=label)
        ax.set_ylabel("Endpoint")
        ax.set_xlabel("Index")
        ax.set_xticklabels([f"{int(x)/1000:.0f}k" for x in ax.get_xticks()])
        return ax

    def plot_yield(
        self,
        endpoint,
        threshold,
        label=None,
        window=100,
        extrapolate=True,
        scaffold=False,
        chemistry_filters_basic=False,
        chemistry_filter_target=False,
    ):
        """
        Plot the yield according to some endpoint and threshold, example, ratio of molecules with reward > 0.8.
        """
        if label is None:
            label = endpoint

        if endpoint not in self.scores.columns:
            print(f"Couldn't find endpoint {endpoint} for plotting")
            return

        tyield, x, y = self.tyield_auc(
            endpoint=endpoint,
            threshold=threshold,
            window=window,
            extrapolate=extrapolate,
            scaffold=scaffold,
            basic_filter=chemistry_filters_basic,
            target_filter=chemistry_filter_target,
            return_trajectory=True,
        )

        ax = sns.lineplot(x=x, y=y, palette="husl", label=label)
        ax.set_ylabel("Yield")
        ax.set_xlabel("Index")
        ax.set_xticklabels([f"{int(x)/1000:.0f}k" for x in ax.get_xticks()])
        return ax

    def _get_chemistry(self, tdf, n=5, scaffold=False, selection=None):
        mol_key = "smiles" if not scaffold else "scaffold"
        # If diverse or random
        if selection:
            if selection == "diverse":
                mols = maxmin_picker(dataset=tdf[mol_key].tolist(), n=n)
                mols = [canonic_smiles(m) for m in mols]
            elif selection == "range":
                idxs = np.linspace(0, len(tdf) - 1, n).astype(int)
                mols = tdf[mol_key].iloc[idxs].tolist()
            elif selection == "random":
                mols = tdf[mol_key].sample(n).tolist()
            elif selection == "similar":
                assert (
                    self.target_smiles
                ), "Target SMILES are needed to identify similar chemistry"
                # Calculate fps
                ECFP = partial(Fingerprints.ECFP4, nBits=1024)
                query_fps = [
                    fp for fp in mapper(self.n_jobs)(ECFP, tdf[mol_key].tolist())
                ]
                ref_fps = [
                    fp
                    for fp in mapper(self.n_jobs)(
                        ECFP,
                        self.target_smiles if not scaffold else self.target_scaffolds,
                    )
                ]
                # Calculate SNNs
                tdf["SNNs"] = [
                    DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
                    for qfp in query_fps
                ]
                # Sort by SNN
                mols = (
                    tdf.sort_values(by="SNNs", ascending=False)[mol_key]
                    .iloc[:n]
                    .tolist()
                )
            else:
                raise ValueError(
                    f"Unknown type of selection method {selection}, please select out of [diverse, random, similar]"
                )
        else:  # Top
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
        bad_only=False,
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
            tdf = self.filter(
                basic=chemistry_filters_basic, target=chemistry_filters_target
            )

        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)

        # If scaffold, drop duplicate scaffolds (largest score is kept)
        if scaffold:
            tdf = tdf.drop_duplicates(subset="scaffold")

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
        bad_only=False,
    ):
        """
        The same as get_chemistry but repeated over the course of de novo generation
        """
        tdf = self.scores
        # Chemisry Filters
        if bad_only:
            import pdb

            pdb.set_trace()
            bad_idxs = [
                i
                for i in range(len(tdf))
                if i
                not in self.chemistry_filter.filter_molecules(
                    tdf.smiles.tolist(), basic=True, target=False
                )
            ]
            tdf = tdf.iloc[bad_idxs]
        else:
            tdf = self.filter(
                basic=chemistry_filters_basic, target=chemistry_filters_target
            )

        # Sort by endpoint
        tdf = tdf.sort_values(by=endpoint, ascending=False)

        # If scaffold, drop duplicate scaffolds (largest score is kept)
        if scaffold:
            tdf = tdf.drop_duplicates(subset="scaffold")

        trajectory_mols = []
        for idx in range(0, len(tdf), window):
            trajectory_mols.append(
                self._get_chemistry(
                    tdf.iloc[idx : idx + window],
                    n=n,
                    scaffold=scaffold,
                    selection=selection,
                )
            )
        return trajectory_mols
