import unittest
from unittest.mock import patch

from database.jobs import insert_jobs, get_by_distance
from database.models import Job
from database.setup import get_session_factory, init_db

class TestJobs(unittest.TestCase):

  def setUp(self) -> None:
    # Keep one engine alive for the whole test. A new SQLite in-memory engine
    # would create a different, empty database.
    self.engine = init_db("sqlite:///:memory:")
    self.session_factory = get_session_factory(self.engine)

    self.open_session_patcher = patch(
      "database.jobs.open_session",
      side_effect=self.session_factory,
    )
    self.open_session_patcher.start()

    self.addCleanup(self.engine.dispose)
    self.addCleanup(self.open_session_patcher.stop)

  def test_method_has_no_jobs(self) -> None:
    # setup
    test_hash = 0b10110111 # 183
    job = Job(filename="job.in", method="m", normalization=set(), simhash32=test_hash)
    insert_jobs(self.session_factory, job)

    # execution
    ref_hash = 0b101000111 # 327
    result = get_by_distance(ref_hash, "not_m")

    # asserts
    self.assertIsNone(result)

  def test_smallest_hamming_distance(self) -> None:
    # setup
    far_hash = 0b001001001 # 73, distance 3
    jobA = Job(filename="a.in", method="lammps_in_files", normalization=set(),
      simhash32=far_hash)

    super_far_hash = 0b111111010 # 506, distance 9
    jobB = Job(filename="b.in", method="lammps_in_files", normalization=set(),
      simhash32=super_far_hash)

    close_hash = 0b000000111 # 7, distance 1
    jobC = Job(filename="c.in", method="lammps_in_files", normalization=set(),
      simhash32=close_hash)

    insert_jobs(self.session_factory, jobA, jobB, jobC)

    # execution
    ref_hash = 0b000000101 # 5
    result = get_by_distance(ref_hash, "lammps_in_files")

    # gold
    gold = "c.in"

    # asserts
    self.assertIsNotNone(result)
    self.assertEqual(result.filename, gold)

  def test_ignores_closer_jobs_from_other_methods(self) -> None:
    # setup
    far_hash = 0b01001001 # 73, distance 7
    jobA = Job(filename="a.in", method="lammps_in_files", normalization=set(),
      simhash32=far_hash)

    close_hash = 0b10110110 # 182, distance 1
    jobB = Job(filename="b.in", method="cmake_cache_variables", normalization=set(),
      simhash32=close_hash)

    insert_jobs(self.session_factory, jobA, jobB)

    # execution
    ref_hash = 0b10110111 # 183
    result = get_by_distance(ref_hash, "lammps_in_files")

    # gold
    gold = "a.in"

    # asserts
    self.assertIsNotNone(result)
    self.assertEqual(result.filename, gold)
