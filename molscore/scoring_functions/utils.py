from typing import Union
import subprocess
import threading
import os
import signal
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem import AllChem as Chem


class timedThread(object):
    """
    Subprocess wrapped into a thread to add a more well defined timeout, use os to send a signal to all PID in
    group just to be sure... (nothing else was doing the trick)
    """

    def __init__(self, timeout):
        self.cmd = None
        self.timeout = timeout
        self.process = None

    def run(self, cmd):
        self.cmd = cmd.split()
        def target():
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self.process.communicate()
            return

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self.timeout)
        if thread.is_alive():
            print('Process timed out...')
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        return


class timedSubprocess(object):
    """
    Currently used
    """
    def __init__(self, timeout, shell=False):
        self.cmd = None
        self.timeout = timeout
        assert isinstance(self.timeout, float)
        self.shell = shell
        self.process = None

    def run(self, cmd):
        if not self.shell:
            self.cmd = cmd.split()
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.cmd = cmd
            self.process = subprocess.Popen(self.cmd, shell=self.shell,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print('Process timed out...')
            if not self.shell:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.kill()
        return


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    return


def get_mol(mol: Union[str, Chem.rdchem.Mol]):
    """
    Get RDkit mol
    :param mol:
    :return: RDKit Mol
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        raise TypeError

    if not mol:
        mol = None

    return mol


def charge_counts(mol: Union[str, Chem.rdchem.Mol]):
    """
    Count the charges based on SMILES, correct for valence separated charges e.g., Nitro
    :param mol: An rdkit mol or str
    :return: Net charge, positive charge, negative charge
    """
    # SMARTS pattern to find single charges on atoms
    charge_pattern = Chem.MolFromSmarts("[+1!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")

    # Accept smiles / mol
    mol = get_mol(mol)
    if mol is None:
        return 0, 0, 0

    # Count charges
    positive_charge = 0
    negative_charge = 0
    at_matches = mol.GetSubstructMatches(charge_pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            if chg > 0:
                positive_charge += chg
            else:
                negative_charge += chg

    net_charge = positive_charge + negative_charge

    return net_charge, positive_charge, negative_charge


def consecutive_rotatable_bonds(mol: Union[str, Chem.rdchem.Mol], include_ring_connections: bool = True):
    """
    Calculate the consecutive rotatable bonds in a molecule, correcting for amides and esters.
    :param mol: Molecule as SMILES or RDKit Mol
    :param include_ring_connections: Whether to include ring connections (will only be of length 1).
    :return: List with rotatable bonds grouped by conesecutive connections
    """
    rotatable_chains = []
    # !D1 More than one bond (exclude terminal atoms)
    # - Single alihpatic bond between two atoms
    # At least one of them isn't in a ring
    rb_patt = Chem.MolFromSmarts('[*!R!D1]-[*!D1]')
    amide_patt = Chem.MolFromSmarts('[NX3][CX3](=[OX1])')
    ester_patt = Chem.MolFromSmarts('[OX2][CX3](=[OX1])')

    rb_matches = mol.GetSubstructMatches(rb_patt)
    amide_matches = mol.GetSubstructMatches(amide_patt)
    ester_matches = mol.GetSubstructMatches(ester_patt)

    for ai, aj in rb_matches:

        # Correct for amides & Esters
        if any([(ai in amide) and (aj in amide) for amide in amide_matches]): continue
        if any([(ai in ester) and (aj in ester) for ester in ester_matches]): continue

        # If either atom found in a set add bonded atom to that set
        for chain in rotatable_chains:
            if ai in chain:
                chain.add(aj)
                break
            elif aj in chain:
                chain.add(ai)
                break
            else:
                pass

        # Both atoms aren't in any chain, add as a new set
        if all([ai not in chain for chain in rotatable_chains]) and all(
                [aj not in chain for chain in rotatable_chains]):
            rotatable_chains.append(set([ai, aj]))

    if include_ring_connections:
        # Single bond between two ring atoms
        rb_rc_patt = Chem.MolFromSmarts('[R!D1]-[R!D1]')
        rb_rc_matches = mol.GetSubstructMatches(rb_rc_patt)
        ring_info = mol.GetRingInfo()

        for ri, rj in rb_rc_matches:

            # If they're both in the same ring ignore
            if any([(ri in ring) and (rj in ring) for ring in ring_info.AtomRings()]):
                continue
            else:
                rotatable_chains.append(set([ri, rj]))

    return rotatable_chains


def max_consecutive_rotatable_bonds(mol: Union[str, Chem.rdchem.Mol], include_ring_connections: bool = True):
    """
    Calculate the consecutive rotatable bonds in a molecule, correcting for amides and esters.
    :param mol: Molecule as SMILES or RDKit Mol
    :param include_ring_connections: Whether to include ring connections (will only be of length 1).
    :return: List with rotatable bonds grouped by conesecutive connections
    """
    rotatable_chains = consecutive_rotatable_bonds(mol=mol, include_ring_connections=include_ring_connections)

    try:
        max_chain_length = len(sorted(rotatable_chains, key=lambda x: len(x))[-1]) - 1
    except IndexError:
        max_chain_length = 0
    return max_chain_length
