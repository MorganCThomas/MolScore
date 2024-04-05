import os
import numpy as np
from typing import Union

import chemprop as cp


class ChemPropModel:
    """
    Score structures by loading a pre-trained chemprop model and return the predicted values
    """
    return_metrics = ['pred_proba']

    def __init__(self, prefix: str, model_path: Union[str, os.PathLike], no_cuda: bool = False, device: int = 0, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model directory
        :param no_cuda: Whether to use cuda
        :param device: Which gpu device to use (if using cuda)
        :param kwargs:
        """
        self.prefix = prefix
        self.prefix = prefix.replace(" ", "_")
        self.model_path = model_path

        # Set chemprop args
        self.args = cp.args.PredictArgs()
        self.args.checkpoint_dir = self.model_path
        self.args.no_cuda = no_cuda
        self.args.gpu = device
        self.args.checkpoint_paths = cp.args.get_checkpoint_paths(
            checkpoint_path=self.args.checkpoint_path,
            checkpoint_paths=self.args.checkpoint_paths,
            checkpoint_dir=self.args.checkpoint_dir,
        )
        cp.data.data.set_cache_mol(not self.args.no_cache_mol)

        if self.args.empty_cache:
            cp.data.data.empty_cache()

        # Load training args
        self.train_args = cp.utils.load_args(self.args.checkpoint_paths[0])
        self.num_tasks, self.task_names = self.train_args.num_tasks, self.train_args.task_names
        cp.utils.update_prediction_args(predict_args=self.args, train_args=self.train_args)

        # Set features
        if self.args.atom_descriptors == 'feature':
            cp.features.featurization.set_extra_atom_fdim(self.train_args.atom_features_size)

        if self.args.bond_features_path is not None:
            cp.features.featurization.set_extra_bond_fdim(self.train_args.bond_features_size)

        # set explicit H option and reaction option
        cp.features.featurization.set_explicit_h(self.train_args.explicit_h)
        cp.features.featurization.set_reaction(self.train_args.reaction, self.train_args.reaction_mode)

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for a ChemProp model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [{'smiles': smi} for smi in smiles]

        # Load smiles
        full_data = cp.data.get_data_from_smiles(
            smiles=[[smi] for smi in smiles],
            skip_invalid_smiles=False,
            features_generator=self.args.features_generator
        )
        # Valid data
        full_to_valid_indices = {}
        valid_index = 0
        for full_index in range(len(full_data)):
            if all(mol is not None for mol in full_data[full_index].mol):
                full_to_valid_indices[full_index] = valid_index
                valid_index += 1

        test_data = cp.data.MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

        # Edge case if empty list of smiles is provided
        if len(test_data) == 0:
            return [None] * len(full_data)

        #print(f'Test size = {len(test_data):,}')

        # Predict with each model individually and sum predictions
        if self.args.dataset_type == 'multiclass':
            sum_preds = np.zeros((len(test_data), self.num_tasks, self.args.multiclass_num_classes))
        else:
            sum_preds = np.zeros((len(test_data), self.num_tasks))

        # Create data loader
        test_data_loader = cp.data.MoleculeDataLoader(
            dataset=test_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )

        # Partial results for variance robust calculation.
        if self.args.ensemble_variance:
            all_preds = np.zeros((len(test_data), self.num_tasks, len(self.args.checkpoint_paths)))

        # Predict
        # print(f'Predicting with an ensemble of {len(self.args.checkpoint_paths)} models')
        for index, checkpoint_path in enumerate(self.args.checkpoint_paths):
            # Load model and scalers
            model = cp.utils.load_checkpoint(checkpoint_path, device=self.args.device)
            scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = cp.utils.load_scalers(checkpoint_path)

            # Normalize features
            if self.args.features_scaling or self.train_args.atom_descriptor_scaling or self.train_args.bond_feature_scaling:
                test_data.reset_features_and_targets()
                if self.args.features_scaling:
                    test_data.normalize_features(features_scaler)
                if self.train_args.atom_descriptor_scaling and self.args.atom_descriptors is not None:
                    test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
                if self.train_args.bond_feature_scaling and self.args.bond_features_size > 0:
                    test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            # Make predictions
            model_preds = cp.train.predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            sum_preds += np.array(model_preds)
            if self.args.ensemble_variance:
                all_preds[:, :, index] = model_preds

        # Ensemble predictions
        avg_preds = sum_preds / len(self.args.checkpoint_paths)
        avg_preds = avg_preds.tolist()

        # if self.args.ensemble_variance:
        #    all_epi_uncs = np.var(all_preds, axis=2)
        #    all_epi_uncs = all_epi_uncs.tolist()

        # Get prediction column names
        # if self.args.dataset_type == 'multiclass':
        #    task_names = [f'{name}_class_{i}' for name in self.task_names
        #                  for i in range(self.args.multiclass_num_classes)]
        # else:
        #    task_names = self.task_names

        for i in range(len(smiles)):
            if i in full_to_valid_indices.keys():
                results[i].update({f'{self.prefix}_pred_proba': avg_preds[full_to_valid_indices[i]][0]})
            else:
                results[i].update({f'{self.prefix}_pred_proba': 0.0})
        return results
