class NeuralNetwork:
    def __init__(self,dna):
        self.fitScoreNorm = 0
        self.fitScore = 0
        self.fit = 0
        self.dna = dna

    def fitness(self):

    def crossover(self,foreignDNAobject):
        crossoverDNA = self.dna.crossover(foreignDNAobject)
        return NeuralNetwork(crossoverDNA)
