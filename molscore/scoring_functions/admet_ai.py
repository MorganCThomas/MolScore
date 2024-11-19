import logging

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("admet_ai")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class ADMETAI(BaseServerSF):
    return_metrics = [
        "AMES",
        "AMES_drugbank_approved_percentile",
        "BBB_Martins",
        "BBB_Martins_drugbank_approved_percentile",
        "Bioavailability_Ma",
        "Bioavailability_Ma_drugbank_approved_percentile",
        "CYP1A2_Veith",
        "CYP1A2_Veith_drugbank_approved_percentile",
        "CYP2C19_Veith",
        "CYP2C19_Veith_drugbank_approved_percentile",
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2C9_Substrate_CarbonMangels_drugbank_approved_percentile",
        "CYP2C9_Veith",
        "CYP2C9_Veith_drugbank_approved_percentile",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP2D6_Substrate_CarbonMangels_drugbank_approved_percentile",
        "CYP2D6_Veith",
        "CYP2D6_Veith_drugbank_approved_percentile",
        "CYP3A4_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels_drugbank_approved_percentile",
        "CYP3A4_Veith",
        "CYP3A4_Veith_drugbank_approved_percentile",
        "Caco2_Wang",
        "Caco2_Wang_drugbank_approved_percentile",
        "Carcinogens_Lagunin",
        "Carcinogens_Lagunin_drugbank_approved_percentile",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Hepatocyte_AZ_drugbank_approved_percentile",
        "Clearance_Microsome_AZ",
        "Clearance_Microsome_AZ_drugbank_approved_percentile",
        "ClinTox",
        "ClinTox_drugbank_approved_percentile",
        "DILI",
        "DILI_drugbank_approved_percentile",
        "HIA_Hou",
        "HIA_Hou_drugbank_approved_percentile",
        "Half_Life_Obach",
        "Half_Life_Obach_drugbank_approved_percentile",
        "HydrationFreeEnergy_FreeSolv",
        "HydrationFreeEnergy_FreeSolv_drugbank_approved_percentile",
        "LD50_Zhu",
        "LD50_Zhu_drugbank_approved_percentile",
        "Lipinski",
        "Lipinski_drugbank_approved_percentile",
        "Lipophilicity_AstraZeneca",
        "Lipophilicity_AstraZeneca_drugbank_approved_percentile",
        "NR-AR",
        "NR-AR-LBD",
        "NR-AR-LBD_drugbank_approved_percentile",
        "NR-AR_drugbank_approved_percentile",
        "NR-AhR",
        "NR-AhR_drugbank_approved_percentile",
        "NR-Aromatase",
        "NR-Aromatase_drugbank_approved_percentile",
        "NR-ER",
        "NR-ER-LBD",
        "NR-ER-LBD_drugbank_approved_percentile",
        "NR-ER_drugbank_approved_percentile",
        "NR-PPAR-gamma",
        "NR-PPAR-gamma_drugbank_approved_percentile",
        "PAMPA_NCATS",
        "PAMPA_NCATS_drugbank_approved_percentile",
        "PPBR_AZ",
        "PPBR_AZ_drugbank_approved_percentile",
        "Pgp_Broccatelli",
        "Pgp_Broccatelli_drugbank_approved_percentile",
        "QED",
        "QED_drugbank_approved_percentile",
        "SR-ARE",
        "SR-ARE_drugbank_approved_percentile",
        "SR-ATAD5",
        "SR-ATAD5_drugbank_approved_percentile",
        "SR-HSE",
        "SR-HSE_drugbank_approved_percentile",
        "SR-MMP",
        "SR-MMP_drugbank_approved_percentile",
        "SR-p53",
        "SR-p53_drugbank_approved_percentile",
        "Skin_Reaction",
        "Skin_Reaction_drugbank_approved_percentile",
        "Solubility_AqSolDB",
        "Solubility_AqSolDB_drugbank_approved_percentile",
        "VDss_Lombardo",
        "VDss_Lombardo_drugbank_approved_percentile",
        "hERG",
        "hERG_drugbank_approved_percentile",
        "hydrogen_bond_acceptors",
        "hydrogen_bond_acceptors_drugbank_approved_percentile",
        "hydrogen_bond_donors",
        "hydrogen_bond_donors_drugbank_approved_percentile",
        "logP",
        "logP_drugbank_approved_percentile",
        "molecular_weight",
        "molecular_weight_drugbank_approved_percentile",
        "stereo_centers",
        "stereo_centers_drugbank_approved_percentile",
        "tpsa",
        "tpsa_drugbank_approved_percentile",
    ]

    def __init__(self, prefix: str = "ADMETAI", env_engine: str = "mamba", **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param env_envine: Environment engine [conda, mamba]
        """
        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="admet_ai",
            env_path=resources.files("molscore.data.models.chemprop.admet_ai").joinpath(
                "admet_ai.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "admet_ai_server.py"
            ),
            server_grace=120,
        )
