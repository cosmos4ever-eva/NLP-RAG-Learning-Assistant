import unittest
from runner import run_experiment
class PublicTest(unittest.TestCase):
    def test_manifest_split(self): self.assertEqual("dev",run_experiment([],"dev",lambda c,s:"")["manifest"]["split"])
if __name__=="__main__": unittest.main()
