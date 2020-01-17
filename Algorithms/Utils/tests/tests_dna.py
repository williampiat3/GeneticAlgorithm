import unittest
from dna_translator import DNA_creator,Continuous_DNA,HybridDNA

class TestDNA(unittest.TestCase):

	def test_discrete(self):
		"""
		Function to test the translation using the discrete approach
		we specifically encode a gene with 3 values to have possible non viable individuals
		(due to the encoding)
		"""
		params = {
		"a":[1,2,3,4],
		"b":[-1,-3,-5]
		}
		translator1 = DNA_creator(params,gray_code=False)
		#testing if generator initialize viable individuals
		self.assertTrue(all([translator1.is_dna_viable(translator1.generate_random()) for _ in range(1000)]))
		#testing the blind spot in the binary coding
		self.assertFalse(translator1.is_dna_viable([1,0,1,1]))

		translator2 = DNA_creator(params,gray_code=True)
		#testing if generator initialize viable individuals
		self.assertTrue(all([translator2.is_dna_viable(translator2.generate_random()) for _ in range(1000)]))
		#testing blind spot in gray encoding
		self.assertFalse(translator2.is_dna_viable([1,0,1,0]))




if __name__ == "__main__":
	unittest.main()
