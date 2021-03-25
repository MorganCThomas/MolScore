"""PDB parser

This module contains functions that can read PDB files and return a
JSON representation of the structural data."""

import re
import json
import os
import tempfile
import gzip
from shutil import copy2
import numpy as np
import matplotlib as mpl

import parmed as pmd
from rdkit import Chem


def sdf_to_dict(sdf_path, seed_index=0):
    """
    dict format
    {"atoms": [{"name": "N", "chain": "A", "positions": [15.407, -8.432, 6.573], "residue_index": 1,
     "element": "N", "residue_name": "GLY1", "serial": 0}, ...],
     "bonds": [{"atom2_index": 0, "atom1_index": 4}, ...]}
     # Bonds absolutely refers to atom list index, where if theres a keyerror
     # i.e. no corresponding atom, throws b or y not available error
     # However, doesn't render unless the atom serial number doesn't match up with atom index ...
    :param sdf_path:
    :param seed_index:
    :return:
    """
    if isinstance(sdf_path, str):

        # Load with rdkit
        if 'gz' in os.path.basename(sdf_path):
            with gzip.open(sdf_path) as f:
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(f, removeHs=False)
                mol = None
                # Just take first mol
                for i, m in enumerate(supp):
                    if i == 0:
                        mol = m
                    else:
                        break
        else:
            supp = Chem.rdmolfiles.ForwardSDMolSupplier(sdf_path, removeHs=False)
            mol = None
            # Just take first mol
            for i, m in enumerate(supp):
                if i == 0:
                    mol = m
                else:
                    break
        if mol is None:
            return {"atoms": [], "bonds": []}

        # Use parmed to read the bond information from temp file
        st = pmd.rdkit.load_rdkit(mol)
        st_df = st.to_dataframe()

        atoms = []
        for index, row in st_df.iterrows():
            d = {}
            d['serial'] = index + seed_index
            d['name'] = row['name']
            d['element'] = re.findall("[A-Z]", row['name'])[0]
            d['chain'] = 'L'
            d['residue_index'] = 1
            d['residue_name'] = os.path.basename(sdf_path).split(".")[0]
            d['positions'] = [row['xx'], row['xy'], row['xz']]
            atoms.append(d)

        bonds = []
        # Create list of bonds using the parmed module
        for i in range(len(st.bonds)):
            bondpair = st.bonds[i].__dict__
            bonds.append({
                'atom1_index': bondpair['atom1'].idx + seed_index,
                'atom2_index': bondpair['atom2'].idx + seed_index
            })
        return {"atoms": atoms, "bonds": bonds}

    else:
        assert isinstance(sdf_path, list), "Only accepts path or list of paths"
        atoms = []
        bonds = []
        sdf_index = 0
        for i, p in enumerate(sdf_path):
            # Load with rdkit
            if 'gz' in os.path.basename(p):
                with gzip.open(p) as f:
                    supp = Chem.rdmolfiles.ForwardSDMolSupplier(f, removeHs=False)
                    mol = None
                    for i, m in enumerate(supp):
                        if i == 0:
                            mol = m
                        else:
                            break
            else:
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(p, removeHs=False)
                mol = None
                for i, m in enumerate(supp):
                    if i == 0:
                        mol = m
                    else:
                        break
            if mol is None:
                continue

            # Use parmed to read the bond information from temp file
            st = pmd.rdkit.load_rdkit(mol)
            st_df = st.to_dataframe()

            for index, row in st_df.iterrows():
                d = {}
                d['serial'] = index + seed_index + sdf_index
                d['name'] = row['name']
                d['element'] = re.findall("[A-Z]", row['name'])[0]
                d['chain'] = 'L'
                d['residue_index'] = i+1
                d['residue_name'] = os.path.basename(p).split(".")[0]
                d['positions'] = [row['xx'], row['xy'], row['xz']]
                atoms.append(d)

            # Create list of bonds using the parmed module
            for i in range(len(st.bonds)):
                bondpair = st.bonds[i].__dict__
                bonds.append({
                    'atom1_index': bondpair['atom1'].idx + seed_index + sdf_index,
                    'atom2_index': bondpair['atom2'].idx + seed_index + sdf_index
                })

            # Update sdf_index i.e. number of atoms in sdf + seed index
            sdf_index += len(st_df)

        return {"atoms": atoms, "bonds": bonds}


def pdb_to_dict(pdb_path):
    """
    dict format
    {"atoms": [{"name": "N", "chain": "A", "positions": [15.407, -8.432, 6.573], "residue_index": 1,
     "element": "N", "residue_name": "GLY1", "serial": 0}, ...],
     "bonds": [{"atom2_index": 0, "atom1_index": 4}, ...]}
    :param pdb_path:
    :return: dict, max_index
    """
    # Use parmed to read the bond information from temp file
    st = pmd.load_file(pdb_path)
    st_df = st.to_dataframe()
    atoms = []
    max_index = 0
    for index, row in st_df.iterrows():
        d = {}
        d['serial'] = index
        d['name'] = row['name']
        d['element'] = row['name'][0]  # Hoping we have no two letter elements in pdb ... Shit!
        d['chain'] = row['chain']
        d['residue_index'] = row['resid']
        d['residue_name'] = row['resname']
        d['positions'] = [row['xx'], row['xy'], row['xz']]
        atoms.append(d)
        max_index = max(max_index, row['number'])

    bonds = []
    # Create list of bonds using the parmed module
    for i in range(len(st.bonds)):
        bondpair = st.bonds[i].__dict__
        bonds.append({
            'atom1_index': bondpair['atom1'].idx,
            'atom2_index': bondpair['atom2'].idx
        })
    return {"atoms": atoms, "bonds": bonds}, max_index


def write_style_dict(model_dict):
    """
    Format dict: {"0": {"color": "#...", "visualization_type": "cartoon/stick/sphere"}}
    :param model_dict:
    :return:
    """
    def get_color(atom_dict, carbon=0):
        carbon_colors = ['#44546A'] + [mpl.colors.to_hex(c) for c in mpl.pyplot.get_cmap('Set3').colors]
        colors = {'H': '#F0F8FF', 'N': '#00008B', 'C': carbon_colors[carbon], 'O': '#B22222', 'S': '#FFD700',
                  'F': '#7FFFD4', 'Cl': '#006400', 'Br': '#B8860B'}
        return colors[atom_dict['element']]

    def euclid(pos1, pos2):
        dist = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
        return dist

    def find_bonding_atoms(model_dict, atom_index):
        allbonds = model_dict['bonds']
        bonding_bonds = [b for b in allbonds if atom_index in b.values()]
        bonding_atom_indexes = []
        for bond in bonding_bonds:
            bonding_atoms = list(bond.values())
            bonding_atoms.remove(atom_index)
            bonding_atom_indexes += bonding_atoms
        bonding_atoms = [model_dict['atoms'][bi] for bi in bonding_atom_indexes]
        return bonding_atoms

    ligand_positions = [atom['positions'] for atom in model_dict['atoms'] if atom['chain'] == 'L']
    ligands = set()

    style_dict = {}
    carbon = 0
    residue_name = None
    for i, atom in enumerate(model_dict['atoms']):
        key = str(i)
        if atom['chain'] != 'L':
            # Check if it's with 6 A of ligand any ligand
            if any([euclid(atom['positions'], lig_pos) < 8 for lig_pos in ligand_positions])\
                    and (atom['element'] != 'H'):
                style_dict[key] = {'color': get_color(atom, carbon=0),
                                   'visualization_type': 'stick'}
            else:
                style_dict[key] = {'color': get_color(atom, carbon=0),
                                   'visualization_type': 'cartoon'}
        else:
            # Cycle through carbon colours based on residue names
            if atom['residue_name'] != residue_name: carbon += 1
            if carbon == 13: carbon = 1

            if atom['element'] == 'H':
                bonding_atoms = find_bonding_atoms(model_dict, i)
                if any([a['element'] in ['O', 'N', 'S'] for a in bonding_atoms]):
                    style_dict[key] = {'color': get_color(atom, carbon=carbon),
                                       'visualization_type': 'stick'}
                else:
                    style_dict[key] = {'color': get_color(atom, carbon=carbon),
                                       'visualization_type': 'cartoon'}
            else:
                style_dict[key] = {'color': get_color(atom, carbon=carbon),
                                   'visualization_type': 'stick'}
        # Save most recent residue name
        residue_name = atom['residue_name']

    return style_dict


def pdb_to_json(pdb_path):
    """
    Parse the protein data bank (PDB) file to generate
    input modelData

    @param pdb_path
    Name of the biomolecular structure file in PDB format

    Json format
    {"atoms": [{"name": "N", "chain": "A", "positions": [15.407, -8.432, 6.573], "residue_index": 1,
     "element": "N", "residue_name": "GLY1", "serial": 0}, ...],
     "bonds": [{"atom2_index": 0, "atom1_index": 4}, ...]}

    """

    # Create local copy of temp file
    copy2(pdb_path, './tmp.pdb')

    # Use parmed to read the bond information from temp file
    top = pmd.load_file('tmp.pdb')

    # Remove the created temp file
    os.remove('tmp.pdb')

    # Read PDB file to create atom/bond information
    with open(pdb_path, 'r') as infile:
        # store only non-empty lines
        lines = [l.strip() for l in infile if l.strip()]

    # Initialize all variables
    var_nchains = []
    serial = []
    atm_name = []
    res_name = []
    chain = []
    res_id = []
    positions = []
    occupancy = []
    temp_factor = []
    atom_type = []
    ct = 0

    datb = {
        'atoms': [],
        'bonds': []
    }

    # Variables that store the character positions of different
    # parameters from the molecule PDB file
    serialpos = [6, 11]
    atm_namepos = [12, 16]
    r_namepos = [17, 20]
    chainpos = [21, 22]
    r_idpos = [22, 26]
    xpos = [30, 38]
    ypos = [38, 46]
    zpos = [46, 54]
    occupos = [54, 60]
    bfacpos = [60, 66]
    atm_typepos = [77, 79]

    for l in lines:
        line = l.split()
        if "ATOM" in line[0] or "HETATM" in line[0]:
            serial.append(int(l[serialpos[0]:serialpos[1]]))
            atm_name.append(l[atm_namepos[0]:atm_namepos[1]].strip())
            val_r_name = l[r_namepos[0]:r_namepos[1]].strip()
            res_name.append(val_r_name)
            chain_val = l[chainpos[0]:chainpos[1]].strip()
            chain.append(chain_val)
            if chain_val not in var_nchains:
                var_nchains.append(chain_val)
            val_r_id = int(l[r_idpos[0]:r_idpos[1]])
            res_id.append(val_r_id)
            x = float(l[xpos[0]:xpos[1]])
            y = float(l[ypos[0]:ypos[1]])
            z = float(l[zpos[0]:zpos[1]])
            positions.append([x, y, z])
            occupancy.append(l[occupos[0]:occupos[1]].strip())
            temp_factor.append(l[bfacpos[0]:bfacpos[1]].strip())
            atom_type.append(l[atm_typepos[0]:atm_typepos[1]].strip())
            ct += 1

    # Create list of atoms
    tmp_res = res_id[0]
    resct = 1
    for i in range(len(chain)):  # pylint: disable=consider-using-enumerate
        if tmp_res != res_id[i]:
            tmp_res = res_id[i]
            resct += 1
        datb['atoms'].append({
            "name": atm_name[i],
            "chain": chain[i],
            "positions": positions[i],
            "residue_index": resct,
            "element": atom_type[i],
            "residue_name": res_name[i] + str(res_id[i]),
            "serial": i,
        })

    # Create list of bonds using the parmed module
    for i in range(len(top.bonds)):
        bondpair = top.bonds[i].__dict__
        atom1 = re.findall(r"\[(\d+)\]", str(bondpair['atom1']))
        atom2 = re.findall(r"\[(\d+)\]", str(bondpair['atom2']))
        datb['bonds'].append({
            'atom2_index': int(atom1[0]),
            'atom1_index': int(atom2[0])
        })

    return json.dumps(datb)


def read_xyz(datapath_or_datastring,
             is_datafile=True):
    """
    Read data in .xyz format, from either a file or a raw string.

    :param (string) datapath_or_datastring: Either the path to the XYZ file (can be relative
                                            or absolute), or a string corresponding to the content
                                            of an XYZ file (including newline characters).
    :param (bool, optional) is_datafile: Either True (default) if passing the filepath to the data,
                                         or False if passing a string of raw data.

    :rtype (list): A list of the atoms in the order that
                   they appear on the file, stored in
                   objects with keys "symbol", "x", "y",
                   and "z".
    """

    # ensure required argument is a string
    err_msg = 'Please pass either the filepath to the data, or the data as a string.'
    assert isinstance(datapath_or_datastring, str), err_msg

    atoms = []

    # open file if given a path
    if is_datafile:
        with open(datapath_or_datastring, 'r') as f:
            lines = f.readlines()

    # or read the raw string
    else:
        lines = datapath_or_datastring.split('\n')

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tf:
        tf.write('\n'.join(lines))

    with open(tf.name, 'r'):
        for line in lines:
            # each line in an xyz file contains the symbol for the atom at
            # that position, as well as the x, y, and z coordinates of the
            # atom separated by spaces
            # the coordinates can have signs (+/-), and decimals (.).
            # an example line in a xyz file:
            # C    +0.125    -1.032    +2.000
            r = re.search(
                r'^\s*([\w]+)\s+([\w\.\+\-]+)\s+([\w\.\+\-]+)\s+([\w\.\+\-]+)\s*',
                line)

            # pass if the line does not contain this information
            if r is None or len(r.groups()) < 4:
                continue

            atom = {
                'symbol': r.group(1),
                'x': float(r.group(2)),
                'y': float(r.group(3)),
                'z': float(r.group(4))
            }

            atoms.append(atom)

    return atoms

