import logging

logger = logging.getLogger(__name__)

#################### List of all scoring functions ####################
all_scoring_functions = []

try:
    from molscore.scoring_functions.descriptors import (
        LinkerDescriptors,
        MolecularDescriptors,
        RDKitDescriptors,
    )

    all_scoring_functions.extend(
        [MolecularDescriptors, RDKitDescriptors, LinkerDescriptors]
    )
except Exception as e:
    logger.warning(f"Descriptors: currently unavailable due to the following: {e}")


try:
    from molscore.scoring_functions.molskill import MolSkill

    all_scoring_functions.append(MolSkill)
except Exception as e:
    logger.warning(f"MolSkill: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.isomer import Isomer

    all_scoring_functions.append(Isomer)
except Exception as e:
    logger.warning(f"Isomer: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.silly_bits import SillyBits

    all_scoring_functions.append(SillyBits)
except Exception as e:
    logger.warning(f"SillyBits: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.similarity import (
        MolecularSimilarity,
        LevenshteinSimilarity,
        TanimotoSimilarity,
    )

    all_scoring_functions.extend([MolecularSimilarity, TanimotoSimilarity, LevenshteinSimilarity])
except Exception as e:
    logger.warning(f"MolecularSimilarity: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.applicability_domain import ApplicabilityDomain

    all_scoring_functions.append(ApplicabilityDomain)
except Exception as e:
    logger.warning(f"ApplicabilityDomain: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.chemistry_filters import ChemistryFilter

    all_scoring_functions.append(ChemistryFilter)
except Exception as e:
    logger.warning(f"ChemistryFilter: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.substructure_filters import SubstructureFilters

    all_scoring_functions.append(SubstructureFilters)
except Exception as e:
    logger.warning(f"SubstructureFilters: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.substructure_match import SubstructureMatch

    all_scoring_functions.append(SubstructureMatch)
except Exception as e:
    logger.warning(f"SubstructureMatch: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.bloom_filter import BloomFilter

    all_scoring_functions.append(BloomFilter)
except Exception as e:
    logger.warning(f"BloomFilter: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.reaction_filter import (
        DecoratedReactionFilter,
        SelectiveDecoratedReactionFilter,
    )

    all_scoring_functions.extend(
        [DecoratedReactionFilter, SelectiveDecoratedReactionFilter]
    )
except Exception as e:
    logger.warning(f"ReactionFilter: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.rascore_xgb import RAScore_XGB

    all_scoring_functions.append(RAScore_XGB)
except Exception as e:
    logger.warning(f"RAScore: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.aizynthfinder import AiZynthFinder

    all_scoring_functions.append(AiZynthFinder)
except Exception as e:
    logger.warning(f"AiZynthFinder: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.pidgin import PIDGIN

    all_scoring_functions.append(PIDGIN)
except Exception as e:
    logger.warning(f"PIDGIN: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.legacy_qsar import LegacyQSAR

    all_scoring_functions.append(LegacyQSAR)
except Exception as e:
    logger.warning(f"LegacyQSAR: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.sklearn_model import (
        EnsembleSKLearnModel,
        SKLearnModel,
    )

    all_scoring_functions.extend([SKLearnModel, EnsembleSKLearnModel])
except Exception as e:
    logger.warning(f"SKLearnModel: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.chemprop import ChemPropModel

    all_scoring_functions.append(ChemPropModel)
except Exception as e:
    logger.warning(f"chemprop: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.admet_ai import ADMETAI
    
    all_scoring_functions.append(ADMETAI)
except Exception as e:
    logger.warning(f"ADMETAI: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.align3d import Align3D

    all_scoring_functions.append(Align3D)
except Exception as e:
    logger.warning(f"Align3D: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.oedock import OEDock
    from molscore.scoring_functions.rocs import ROCS, GlideDockFromROCS

    all_scoring_functions.extend([ROCS, GlideDockFromROCS, OEDock])
except Exception as e:
    logger.warning(f"OpenEye functions: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.glide import GlideDock

    all_scoring_functions.append(GlideDock)
except Exception as e:
    logger.warning(f"GlideDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.plants import PLANTSDock

    all_scoring_functions.append(PLANTSDock)
except Exception as e:
    logger.warning(f"PLANTSDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.gold import (
        ASPGOLDDock,
        ChemPLPGOLDDock,
        ChemScoreGOLDDock,
        GOLDDock,
        GoldScoreGOLDDock,
    )

    all_scoring_functions.extend(
        [GOLDDock, ChemPLPGOLDDock, ASPGOLDDock, ChemScoreGOLDDock, GoldScoreGOLDDock]
    )
except Exception as e:
    logger.warning(f"GOLD: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.rdock import rDock

    all_scoring_functions.append(rDock)
except Exception as e:
    logger.warning(f"rDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.smina import SminaDock

    all_scoring_functions.append(SminaDock)
except Exception as e:
    logger.warning(f"SminaDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.gnina import GninaDock

    all_scoring_functions.append(GninaDock)
except Exception as e:
    logger.warning(f"GninaDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.vina import VinaDock

    all_scoring_functions.append(VinaDock)
except Exception as e:
    logger.warning(f"VinaDock: currently unavailable due to the following: {e}")

try:
    from molscore.scoring_functions.external_server import POSTServer

    all_scoring_functions.append(POSTServer)
except Exception as e:
    logger.warning(f"POSTServer: currently unavailable due to the following: {e}")