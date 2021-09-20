import itertools
import logging
import lzma
import pickle

import pytest

FACT_FILES = [
    "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/data.xlsx",
    "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data2/data2.xlsx",
]
TRACE_FILES = [
    "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/Chemical space 1 knowledge.pz"
]


@pytest.fixture(params=itertools.product(FACT_FILES, TRACE_FILES))
def mgr(request):
    from chem_oracle.experiment import ExperimentManager

    with lzma.LZMAFile(request.param[1]) as f:
        knowledge_trace = pickle.load(f)["trace"]

    mgr = ExperimentManager(
        request.param[0],
        fingerprint_radius=3,
        fingerprint_bits=512,
        structural_model=True,
        log_level=logging.DEBUG,
        knowledge_trace=knowledge_trace,
        monitor=False,
    )

    # use a smaller fact set to cut down on runtime
    mgr.reactions_df = mgr.reactions_df[:50]

    return mgr


def test_disruption(mgr):
    mgr.model.condition(mgr.reactions_df, trace=mgr.knowledge_trace)
    mgr.model.condition(mgr.reactions_df, trace=mgr.knowledge_trace)


def test_likelihoods(mgr):
    x = mgr.model.log_likelihoods(mgr.reactions_df, trace=mgr.knowledge_trace)
    y = mgr.model.experiment_likelihoods(mgr.reactions_df, trace=mgr.knowledge_trace)
