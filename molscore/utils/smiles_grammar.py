import nltk

# Enhanced SMILES grammar to handle complex metal coordination environments, nested aromatic systems, and extended ring handling
gram = """
smiles -> chain
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
chain -> chain bond chain

branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
branched_atom -> metal_complex

# Defining atom types
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
atom -> sulfur_aromatic 
atom -> bracketed_atom_symbol 

# Handling transition metals and their different bonding states, charges, hydrogens, and ring closures
bracket_atom -> '[' metal_symbol ']'
bracket_atom -> '[' metal_symbol hcount ']'
bracket_atom -> '[' metal_symbol charge ']'
bracket_atom -> '[' metal_symbol hcount charge ']'
bracket_atom -> '[' metal_symbol ringbond ']'
bracket_atom -> '[' metal_symbol hcount ringbond ']'
bracket_atom -> '[' metal_symbol charge ringbond ']'
bracket_atom -> '[' metal_symbol hcount charge ringbond ']'
bracket_atom -> '[' metal_symbol BB ']'
bracket_atom -> '[' metal_symbol RB ']'
bracket_atom -> '[' metal_symbol RB BB ']'

bracket_atom -> '[' BAI ']'
bracket_atom -> '[' BAI ringbond ']'
bracket_atom -> '[' BAI charge ']'
bracket_atom -> '[' BAI BB ']'

# Handling atoms in brackets directly, e.g., [Sc]
bracketed_atom_symbol -> '[Sc]' 
bracketed_atom_symbol -> '[Sc+]'
bracketed_atom_symbol -> '[Sc+ DIGIT]'

# Handling complex metal-ligand coordination structures
metal_complex -> '[' metal_symbol complex_ligands ']'
metal_complex -> '[' metal_symbol hcount complex_ligands ']'
metal_complex -> '[' metal_symbol charge complex_ligands ']'
metal_complex -> '[' metal_symbol hcount charge complex_ligands ']'
metal_complex -> '[' metal_symbol ringbond complex_ligands ']'
metal_complex -> '[' metal_symbol hcount charge ringbond complex_ligands ']'
metal_complex -> '[' metal_symbol RB BB complex_ligands ']'

complex_ligands -> ligand
complex_ligands -> complex_ligands ligand
ligand -> branch
ligand -> aliphatic_organic
ligand -> aromatic_organic
ligand -> bond ligand  
ligand -> metal_complex 

# Defining 'S' + 'c' explicitly to distinguish from scandium
sulfur_aromatic -> 'S' 'c'

# Defining allowed metal symbols (must be used with brackets)
metal_symbol -> 'Cd' | 'Os' | 'Ti' | 'Rh' | 'Ce' | 'Hg' | 'Cf' | 'Pt' | 'Au' | 'Lu' | 'Cm' | 'Ni' | 'Ho' | 'Nd' | 'Np' | 'Pu' | 'Yb' | 'Tb' | 'Pa' | 'Ag' | 'V' | 'La' | 'U' | 'Ru' | 'Eu' | 'Pd' | 'Zn' | 'Cr' | 'Sm' | 'Am' | 'Dy' | 'Nb' | 'Re' | 'W' | 'Th' | 'Pr' | 'Hf' | '[Sc]' | 'Tm' | 'Mn' | 'Mo' | 'Y' | 'Er' | 'Co' | 'Tc' | 'Gd' | 'Zr' | 'Ta' | 'Fe' | 'Bk' | 'Cu' | 'Ir' | 'Li' | 'Na' | 'Sn' | 'Bi' | 'Sb' | 'K' | 'Al' | 'Rb' | 'Ba' | 'Pb' | 'Mg' | 'Be' | 'Ca' | 'Sr' | 'Cs' | 'In' | 'Tl' 

# Existing rules for organic and inorganic atoms
# Non-metal elements that are not in brackets
aliphatic_organic -> 'B' | 'C' | 'F' | 'H' | 'I' | 'N' | 'O' | 'P' | 'S' | 'Cl' | 'Br' | 'Si' | 'Se' | 'As' | 'Ge' | 'Ga' | 'Te'

aromatic_organic -> 'b' | 'c' | 'n' | 'o' | 'p' | 's' | 'se'

# Original BAI and BAC definitions for atoms with isotopes or chirality
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge

symbol -> aliphatic_organic
symbol -> aromatic_organic

# Isotopes and valency rules
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0'

chiral -> '@' | '@@'
hcount -> 'H' | 'H' DIGIT
charge -> '-' | '-' DIGIT | '-' DIGIT DIGIT | '+' | '+' DIGIT | '+' DIGIT DIGIT

bond -> '-' | '=' | '#' | '/' | '\\' | ':' 

# Extend ringbond to include % followed by double digits
ringbond -> DIGIT | bond DIGIT | DIGIT DIGIT | bond DIGIT DIGIT | '%' DIGIT DIGIT | bond '%' DIGIT DIGIT

RB -> RB ringbond | ringbond
BB -> BB branch | branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
branch -> '(' chain branch ')'
branch -> '(' bond chain branch ')'
branch -> '(' chain branch branch ')'

Nothing -> None
"""

# Form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
