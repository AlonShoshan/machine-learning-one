import random

def getRandCaracter():
    randNum = random.randint(0, ord('z') - ord('a') - 1)
    return chr(randNum + ord('a'))

class DNA:
    def __init__(self, genes=None, mutationFactor=0.01):
        self.mutationFactor = mutationFactor
        self.fitScoreNorm = 0
        self.fitScore = 0
        self.fit = 0
        self.text = ['s','u','c','c','e','s','s']
        if genes is None :
            self.genes = []
            for i in range(len(self.text)):
                self.genes.append(getRandCaracter())
        else :
            self.genes = genes

    def crossover(self, foreignDNAobject):
        newGenes = []
        for i in range(len(self.genes)):
            if random.randint(0, 1) == 1 :
                newGenes.append(self.genes[i])
            else:
                newGenes.append(foreignDNAobject.genes[i])
        return DNA(newGenes,mutationFactor=self.mutationFactor)

    def mutate(self):
        for i in range(len(self.genes)):
            if random.randint(0, 100) < self.mutationFactor*100 :
                self.genes[i] = getRandCaracter()

    def fitness(self):
        self.fit = 0
        for i in range(len(self.genes)):
            if self.genes[i] == self.text[i] :
                self.fit += 1
        return self.fit
