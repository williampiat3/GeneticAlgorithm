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
		#testing if generator initializes viable individuals
		self.assertTrue(all([translator1.is_dna_viable(translator1.generate_random()) for _ in range(1000)]))
		#testing the blind spot in the binary coding
		self.assertFalse(translator1.is_dna_viable([1,0,1,1]))
		#testing translation capacity
		try:
			translator1.dna_to_phen([1,0,1,0])
		except Exception:
			self.fail("translation failed on discrete dna, binary encoding")


		translator2 = DNA_creator(params,gray_code=True)
		#testing if generator initializes viable individuals
		self.assertTrue(all([translator2.is_dna_viable(translator2.generate_random()) for _ in range(1000)]))
		#testing blind spot in gray encoding
		self.assertFalse(translator2.is_dna_viable([1,0,1,0]))
		#testing translation capacity
		try:
			translator2.dna_to_phen([1,0,1,1])
		except Exception:
			self.fail("translation failed on discrete dna, gray code encoding")

		

	def test_continuous(self):
		"""
		Function to test the translation operated by the continuous dna
		the format of the parameters has to be precise as this is the only
		format decoded by the continuous DNA
		"""
		params = {
		"a":{"range":[2,8],"scale":"linear"},
		"b":{"range":[0.1,10],"scale":"log"}
		}

		translator1 = Continuous_DNA(params,stric_interval=False)
		#testing if generator initializes viable individuals
		self.assertTrue(all([translator1.is_dna_viable(translator1.generate_random()) for _ in range(1000)]))
		#testing if the translator accepts genes beyond initial scope
		self.assertTrue(translator1.is_dna_viable([1,1.1]))
		#testing translation capacity
		try:
			translator1.dna_to_phen([1.,1.9])
		except Exception:
			self.fail("translation failed on continuous dna, not strict interval")



		#changing the value of the strictness of the interval
		translator2 = Continuous_DNA(params,stric_interval=True)
		#testing if generator initializes viable individuals
		self.assertTrue(all([translator2.is_dna_viable(translator2.generate_random()) for _ in range(1000)]))
		#testing if translator rejects the individual once interval are strict
		self.assertFalse(translator2.is_dna_viable([1.,1.1]))
		#testing translation capacity
		try:
			translator2.dna_to_phen([1.,0.9])
		except Exception:
			self.fail("translation failed on continuous dna, strict interval")


	def test_hybrid(self):
		"""
		Function to test the HybrisDNA class, it mixes the two other approaches
		"""
		params={}
		params["discrete"] = {
		"a":[1,2,3,4],
		"b":[-1,-3,-5]
		}
		params["continuous"] = {
		"c":{"range":[2,8],"scale":"linear"},
		"d":{"range":[0.1,10],"scale":"log"}
		}
		translator1 = HybridDNA(params,gray_code=False,stric_interval=False)
		#testing if generator initialize viable individuals
		self.assertTrue(all([translator1.is_dna_viable(translator1.generate_random()) for _ in range(1000)]))

		#the genotype here is [discrete_gene,continuous_gen]
		#testing if the translator reject faulty discrete dna
		self.assertFalse(translator1.is_dna_viable([[1,0,1,1],[1,1.1]]))
		#testing if the translator accepts genes beyond initial scope
		self.assertTrue(translator1.is_dna_viable([[1,0,1,0],[1,1.1]]))
		#testing translation capacity
		try:
			translator1.dna_to_phen([[1,0,1,0],[1,1.1]])
		except Exception:
			self.fail("translation failed on hybrid dna, not strict interval, binary encoding")


		translator2 = HybridDNA(params,gray_code=True,stric_interval=True)
		#testing if generator initialize viable individuals
		self.assertTrue(all([translator2.is_dna_viable(translator2.generate_random()) for _ in range(1000)]))

		#the genotype here is [continuous_gene,discrete_gen]
		#testing if the translator rejects continuous part
		self.assertFalse(translator2.is_dna_viable([[1,0,1,1],[1,1.1]]))
		#testing if the translator rejects genes discrete part
		self.assertFalse(translator2.is_dna_viable([[1,0,1,0],[1.,1.]]))

		#testing translation capacity
		try:
			translator2.dna_to_phen([[1,0,1,1],[1.,1.]])
		except Exception:
			self.fail("translation failed on hybrid dna, strict interval, gray encoding")






if __name__ == "__main__":
	unittest.main()
