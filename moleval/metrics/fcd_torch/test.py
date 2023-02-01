from moleval.metrics.fcd_torch import FCD
import unittest


class test_same_output(unittest.TestCase):
    def setUp(self):
        self.set1 = ['Oc1ccccc1-c1cccc2cnccc12',
                     'COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1']
        self.set2 = ['CNC', 'CCCP',
                     'Oc1ccccc1-c1cccc2cnccc12',
                     'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1',
                     'Cc1nc(NCc2ccccc2)no1-c1ccccc1']
        self.output_keras = 52.83132961802335

    def test_output(self):
        fcd = FCD()
        output_pytorch = fcd(self.set1, self.set2)
        diff = abs(self.output_keras-output_pytorch)
        self.assertAlmostEqual(
            output_pytorch, self.output_keras, places=4,
            msg=("Outputs differ. keras={},".format(self.output_keras) +
                 "torch={}. diff is {}".format(output_pytorch, diff))
        )

    def test_one_molecule(self):
        fcd = FCD()
        output_pytorch = fcd(['C'], ['C'])
        self.assertNotEqual(
            output_pytorch, output_pytorch,
            msg=("FCD should return np.nan on invalid situations")
        )

    def test_zero_molecule(self):
        fcd = FCD()
        output_pytorch = fcd([], [])
        self.assertNotEqual(
            output_pytorch, output_pytorch,
            msg=("FCD should return np.nan on invalid situations")
        )

    def test_multiprocess(self):
        fcd = FCD(n_jobs=2)
        output_pytorch = fcd(self.set1, self.set2)
        diff = abs(self.output_keras-output_pytorch)
        self.assertAlmostEqual(
            output_pytorch, self.output_keras, places=4,
            msg=("Outputs differ. keras={},".format(self.output_keras) +
                 "torch={}. diff is {}".format(output_pytorch, diff))
        )


if __name__ == '__main__':
    unittest.main()
