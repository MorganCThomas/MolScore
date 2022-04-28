import os
import unittest
import inspect

from glob import glob


class BaseTests:
    class TestScoringFunction(unittest.TestCase):
        """
        Generic tests to be inherited by specific scoring function tests... based on inheriting
         self.obj, self.inst, self.input & self.output
        """
        output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_out')

        def test_has_return_metrics(self):
            self.assertIsNotNone(getattr(self.obj, 'return_metrics', None))
            self.assertIsInstance(self.obj.return_metrics, list,
                                  'return_metrics should be a list')
            self.assertGreater(len(self.obj.return_metrics), 0,
                               'return_metrics should contain at least one metric')

        def test_return_metric_in_result(self):
            for m in self.obj.return_metrics:
                for o in self.output:
                    with self.subTest('Checking all outputs'):
                        self.assertTrue(any([m in k for k in o.keys()]),
                                        'Metric should be in at least one key for all outputs')

        def test_prefix(self):
            self.assertIn('prefix', inspect.signature(self.obj.__init__).parameters.keys())

        def test_output(self):
            self.assertEqual(len(self.input), len(self.output),
                             'Length of output should be the same as the length of input')
            self.assertIsInstance(self.output, list, 'Output should be a list')
            for o in self.output:
                with self.subTest('Checking all outputs'):
                    self.assertIsInstance(o, dict)
                    self.assertIn('smiles', o.keys())
                    # Check all return metrics are in dict
                    for rm in self.obj.return_metrics:
                        self.assertIn(f'{self.inst.prefix}_{rm}', o.keys(), f"{rm} not in output")
                    # Check all dict keys are in return metrics
                    o.pop('smiles')
                    for k in o.keys():
                        self.assertIn(k.replace(f'{self.inst.prefix}_', ''), self.obj.return_metrics, f"{k} not in return metrics")

    class TestLigandPreparation(unittest.TestCase):
        """
        Generic tests to be inherited by specific ligand preparation protocols based on inheriting
          self.obj, self.inst, self.input & self.output
        """
        output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_out')

        def test_output(self):
            self.assertIsInstance(self.output, tuple)
            out1, out2 = self.output
            self.assertIsInstance(out1, dict)
            self.assertIsInstance(out2, list)
            for i in out1:
                for v in out1[i]:
                    self.assertGreater(len(glob(os.path.join(self.output_directory, f"{i}-{v}_prepared.*"))), 0)

            for f in out2:
                self.assertTrue(os.path.exists(f))