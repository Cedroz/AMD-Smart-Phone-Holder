"""Read and write the portable NumPy voiceprint file."""

import numpy as np

import config


def load_voiceprints(path=config.VOICEPRINTS_PATH):
    with np.load(path, allow_pickle=False) as archive:
        names = archive["speaker_names"].tolist()
        return {name: archive[f"embedding_{name}"].astype(np.float32) for name in names}


def save_voiceprints(voiceprints, path=config.VOICEPRINTS_PATH):
    values = {"speaker_names": np.asarray(list(voiceprints), dtype="U")}
    values.update({f"embedding_{name}": vector for name, vector in voiceprints.items()})
    np.savez(path, **values)