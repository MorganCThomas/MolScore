import nltk

import numpy as np

from . import smiles_grammar


def get_smiles_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index.keys() if len(a) > 1]
    # There are currently 76 double letter entities in the grammar
    # these are their replacement, with no particular meaning
    # they need to be ascii and not part of the SMILES symbol vocabulary
    
    replacements = [
    '!', '?', '$', '&', '*', '~', '_', ';', '.', ',',
    '`', '|', '<', '>', '{', '}', '§', '^', 'a', 'A',
    'z', 'Z', 'm', 'M', 'd', 'D', 'e', 'E', 'g', 'G',
    'j', 'J', 'l', 'L', 'q', 'Q', 'r', 'R', 't', 'T',
    'x', 'X', '"', '´', '˚', 'å', 'Å', 'ø', 'Ø', '¨',
    'ä', 'Ä', 'ö', 'Ö', 'ü', 'Ü', 'á', 'Á', 'é', 'É',
    'í', 'Í', 'ó', 'Ó', 'ú', 'Ú', 'à', 'À', 'è', 'È',
    'ì', 'Ì', 'ò', 'Ò', 'ù', 'Ù', 
    ]

    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except Exception:
                tokens.append(token)
        return tokens
    return tokenize


def encode(smiles):
    GCFG = smiles_grammar.GCFG
    tokenize = get_smiles_tokenizer(GCFG)
    tokens = tokenize(smiles)
    parser = nltk.ChartParser(GCFG)
    try:
        parse_tree = parser.parse(tokens).__next__()
    except StopIteration:
        print(f'Failed to parse {smiles}')
        return None
    productions_seq = parse_tree.productions()
    productions = GCFG.productions()
    prod_map = {}
    for ix, prod in enumerate(productions):
        prod_map[prod] = ix
    indices = np.array([prod_map[prod] for prod in productions_seq], dtype=int)
    return indices


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except Exception:
        return ''

def decode(rule):
    productions = smiles_grammar.GCFG.productions()
    prod_seq = [productions[i] for i in rule]
    return prods_to_eq(prod_seq)
