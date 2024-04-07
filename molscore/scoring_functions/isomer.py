import re

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from scipy.stats import gmean

from molscore.utils.transformation_functions import gauss


class Isomer:
    """
    Score structures according to molecular formula:
         re-implementation of GuacaMol https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.8b00839
    """

    return_metrics = ["isomer_score"]

    def __init__(self, prefix: str, molecular_formula: str, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., C2H6)
        :param molecular_formula: Element count of 1 must be explicit (simplicity hack e.g., C9H10N2O2P1F2Cl1)
        :param kwargs:
        """
        self.prefix = prefix
        self.ref_formula = molecular_formula
        self.ref_elements = self.formula2elements(self.ref_formula)

    @staticmethod
    def formula2elements(formula):
        """
        Use regex to retrieve elements and counts
        """
        components = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

        # Convert matches to the required format
        elements = {}
        for c in components:
            # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
            count = 1 if not c[1] else int(c[1])
            elements[c[0]] = count
        return elements

    def calculate_isomer_score(self, smi):
        """
        Calculate isomer score based on geometric mean of Gaussian modifiers applied to element counts.
        :param smi:
        :return:
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            query_formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
            query_elements = self.formula2elements(query_formula)
            elements = set(self.ref_elements)
            elements.update(query_elements)
            scores = []
            # Add per element scores
            for e in elements:
                scores.append(
                    gauss(
                        x=query_elements[e] if e in query_elements else 0,
                        objective="range",
                        mu=self.ref_elements[e] if e in self.ref_elements else 0,
                        sigma=1,
                    )
                )
            # Add total atoms score
            scores.append(
                gauss(
                    x=sum(query_elements.values()),
                    objective="range",
                    mu=sum(self.ref_elements.values()),
                    sigma=2,
                )
            )
            # Add dummy value for 0's to allow geometric mean
            scores = [s if s != 0 else 1e-6 for s in scores]
            score = gmean(scores)
        else:
            score = 0.0
        return score

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate Isomer scores for a given list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [
            {
                "smiles": smi,
                f"{self.prefix}_{self.return_metrics[0]}": self.calculate_isomer_score(
                    smi
                ),
            }
            for smi in smiles
        ]
        return results


if __name__ == "__main__":
    from molscore.tests import MockGenerator

    mg = MockGenerator()
    iso = Isomer("C12H24", "C12H24")
    iso(mg.sample(5))
