import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
import pandas as pd
import os
import json


def read_rules(rules_file_name):
    """
    Read rules from a JSON file
    :param rules_file_name: JSON file name
    :return: dictionary corresponding to the contents of the JSON file
    """
    with open(rules_file_name) as json_file:
        try:
            rules_dict = json.load(json_file)
            return rules_dict
        except json.JSONDecodeError:
            print(f"Error parsing JSON file {rules_file_name}")
            sys.exit(1)


def write_rules(rule_dict, file_name):
    """
    Write configuration to a JSON file
    :param rule_dict: dictionary with rules
    :param file_name: JSON file name
    :return: None
    """
    ofs = open(file_name, "w")
    ofs.write(json.dumps(rule_dict, indent=4, sort_keys=True))
    print(f"Wrote rules to {file_name}")
    ofs.close()


def default_rule_template(alert_list, file_name):
    """
    Build a default rules template
    :param alert_list: list of alert set names
    :param file_name: output file name
    :return: None
    """
    default_rule_dict = {
        "MW": [0, 500],
        "LogP": [-5, 5],
        "HBD": [0, 5],
        "HBA": [0, 10],
        "TPSA": [0, 200],
        "Rot": [0, 10]
    }
    for rule_name in alert_list:
        if rule_name == "Inpharmatica":
            default_rule_dict["Rule_" + rule_name] = True
        else:
            default_rule_dict["Rule_" + rule_name] = False
    write_rules(default_rule_dict, file_name)


def get_config_file(file_name, environment_variable):
    """
    Read a configuration file, first look for the file, if you can't find
    it there, look in the directory pointed to by environment_variable
    :param file_name: the configuration file
    :param environment_variable: the environment variable
    :return: the file name or file_path if it exists otherwise exit
    """
    if os.path.exists(file_name):
        return file_name
    else:
        config_dir = os.environ.get(environment_variable)
        if config_dir:
            config_file_path = os.path.join(os.path.sep, config_dir, file_name)
            if os.path.exists(config_file_path):
                return config_file_path

    error_list = [f"Could not file {file_name}"]
    if config_dir:
        err_str = f"Could not find {config_file_path} based on the {environment_variable}" + \
                  "environment variable"
        error_list.append(err_str)
    error_list.append(f"Please check {file_name} exists")
    error_list.append(f"Or in the directory pointed to by the {environment_variable} environment variable")
    print("\n".join(error_list))
    sys.exit(1)


class RDFilters:
    def __init__(self, rules_file_name):
        good_name = get_config_file(rules_file_name, "FILTER_RULES_DIR")
        self.rule_df = pd.read_csv(good_name)
        # make sure there wasn't a blank line introduced
        self.rule_df = self.rule_df.dropna()
        self.rule_list = []

    def build_rule_list(self, alert_name_list):
        """
        Read the alerts csv file and select the rule sets defined in alert_name_list
        :param alert_name_list: list of alert sets to use
        :return:
        """
        self.rule_df = self.rule_df[self.rule_df.rule_set_name.isin(alert_name_list)]
        tmp_rule_list = self.rule_df[["rule_id", "smarts", "max", "description"]].values.tolist()
        for rule_id, smarts, max_val, desc in tmp_rule_list:
            smarts_mol = Chem.MolFromSmarts(smarts)
            if smarts_mol:
                self.rule_list.append([smarts_mol, max_val, desc])
            else:
                print(f"Error parsing SMARTS for rule {rule_id}", file=sys.stderr)

    def get_alert_sets(self):
        """
        :return: a list of unique rule set names
        """
        return self.rule_df.rule_set_name.unique()

    def evaluate(self, mol):
        """
        Evaluate structure alerts on a list of SMILES
        :param lst_in: input list of [SMILES, Name]
        :return: list of alerts matched or "OK"
        """
        if mol is None:
            return ['INVALID', -999, -999, -999, -999, -999, -999]
        desc_list = [MolWt(mol), MolLogP(mol), NumHDonors(mol), NumHAcceptors(mol), TPSA(mol),
                     CalcNumRotatableBonds(mol)]
        for row in self.rule_list:
            patt, max_val, desc = row
            if len(mol.GetSubstructMatches(patt)) > max_val:
                return [desc + " > %d" % (max_val)] + desc_list
        return ["OK"] + desc_list