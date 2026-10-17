"""Stand-in for run_scheduler_process in the prespawn tests."""

import os


def run(local_rank, writer, payload):
    writer.send((local_rank, payload, os.environ.get("PRESPAWN_TEST_ENV")))
    writer.close()
