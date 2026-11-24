import unittest
from runner import run_cases
class PublicTest(unittest.TestCase):
    def test_returns_something(self): self.assertTrue(run_cases([{"id":"x"}],lambda c,s:"a"))
if __name__ == "__main__": unittest.main()
