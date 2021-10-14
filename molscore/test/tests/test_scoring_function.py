import unittest


class BaseTests:
    class TestScoringFunction(unittest.TestCase):
        """
        Generic tests to be inherited by specific scoring function tests... based on inheriting
         self.cls, self.inst, self.input & self.output
        """

        def test_has_return_metrics(self):
            self.assertIn('return_metrics', self.cls.__dict__.keys(),
                          'Class is missing \'return_metrics\' attribute')
            self.assertIsInstance(self.cls.return_metrics, list,
                                  'return_metrics should be a list')
            self.assertGreater(len(self.cls.return_metrics), 0,
                               'return_metrics should contain at least one metric')

        def test_return_metric_in_result(self):
            for m in self.cls.return_metrics:
                for o in self.output:
                    with self.subTest('Checking all outputs'):
                        self.assertTrue(any([m in k for k in o.keys()]),
                                        'Metric should be in at least one key for all outputs')

        def test_output(self):
            self.assertEqual(len(self.input), len(self.output),
                             'Length of output should be the same as the length of input')
            for o in self.output:
                with self.subTest('Checking all outputs'):
                    self.assertIn('smiles', o.keys())
