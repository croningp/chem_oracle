if __name__ == "__channelexec__":
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    def _morgan_bits(smiles, radius, nbits):
        mol = Chem.MolFromSmiles(smiles)
        result = np.zeros(nbits)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        result[morgan_fp.GetOnBits()] = 1.0
        return result.tolist()

    while True:
        channel.send(_morgan_bits(*channel.receive()))
else:
    from chem_oracle import rdkit_fns
    import execnet
    import numpy as np

    gw = execnet.makegateway(
        "popen//python=/home/group/miniconda3/envs/rdkit/bin/python"
    )
    ch = gw.remote_exec(rdkit_fns)

    def morgan_bits(smiles: str, radius: int, nbits: int) -> np.ndarray:
        ch.send((smiles, radius, nbits))
        return np.array(ch.receive())
