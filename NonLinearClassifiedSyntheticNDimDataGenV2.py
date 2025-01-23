# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:08:13 2024

@author: Zubair
"""
import matplotlib.pyplot as plt 
import numpy as np
rng = np.random.default_rng()
class SynthNonLinearClassifiedDataGen:
    # We can control the attribute/dimension of data to be generated and also the intersection of the imaginary lines that help us label the data in a non-linear manner
    # The intersection can be left unassigned it will take default value
    # If intersection flag is set to True, it will randomly generate the intersection point , and can supply planes/lines to be used for non-linear class labeling
    # Size is the number of points we want to produce the data for
    def __init__(self, dimensions=2,size=200, intersection=[0.2,0.5],intersection_flag = False, planes=None,chromosome = ""):
        # define how user-supplied data values will be assigned 
        # to each instance variable within the class
        self.dimensions = dimensions
        self.size = size
        self.intersection_flag = intersection_flag
        self.chromosome = chromosome

        if self.intersection_flag:
            self.intersection = 0.6*np.random.random((self.dimensions,))+0.2
        else:
            if len(intersection) == self.dimensions:
                self.intersection = intersection
            else:
                print("The intersection point dimension missmatch with dimension of hyperplanes")
        if str(type(planes)) == "<class 'NoneType'>":
            self.planes = [self.hyper_plane(self.dimensions) for i in range(2)]
        else:
            self.planes = planes

    # To return the HyperPlanes (In case of 2D data, this returns line parameters otherwise hyperplane parameters)
    def get_hyper_planes(self):
        return self.planes

    # Generates synthetic data all attributes/dimensions randomly between 0 and 1
    def generate_data(self,arr = None):
        if str(type(arr)) == "<class 'NoneType'>":
            self.arr = rng.random((self.size, self.dimensions))
        else:
            self.arr = arr
        return self.arr
        
    # Produces the parameters of a hyperplane
    def hyper_plane(self,dimensions = 2):
        A = np.random.random((dimensions,))#rng.random((dimensions,1))
        B = -np.dot(self.intersection,A)
        return A,B
    
    # Classify if a point p is above or below a hyperplane/line using the hyperplane/line parameters
    def classify(self,A,B,p):
        return (np.dot(p,A)+B) > 0
    
    # Classify if a point p is above or below a  Choquet hyperplane/line 
    def classify_choquet(self,p, mu, B, a, b, dimensions = 2):
        return self.generalized_choquet_integral(p, mu, B, a, b) > B
        
    # Generates boolean class value based on the logic if point lies between two lines at a specific quadrant of the intersection of lines/hyperplanes
    def label(self,p):
        class_label = False
        if self.classify(self.planes[0][0],self.planes[0][1],p[0:2]):
            if not self.classify(self.planes[1][0],self.planes[1][1],p[0:2]):
                class_label = True
        return class_label
    
    # Generates boolean class value based on the Choquet integral
    def label_choquet(self,p, dimensions = 2): #(self,p, mu, B, a, b, dimensions = 2):
        mu, B, a, b = self.infer_chromosome(self.chromosome)
        class_label = False
        if self.classify_choquet(p, mu, B, a, b, dimensions):
                class_label = True
        return class_label
    
    # Labels all the dataset at once using Choquet Hyperplane labeling
    def labeler_choquet(self,arr):
        self.class_labels = np.apply_along_axis(self.label_choquet, axis=1, arr=arr)
        return self.class_labels
    
    # Plots the synthetic data along with the class they belong to based on color using Choquet integral
    def spread_plotter_choquet(self,p):
        class_flag = self.label_choquet(p)
        if len(p) !=0:
            if class_flag:
                plt.plot(p[0],p[1],'*g')
            else:
                plt.plot(p[0],p[1],'*b')
    
    # Labels all the dataset at once using lines/hyperplanes for non-linear artificial class labeling
    def labeler(self,arr):
        self.class_labels = np.apply_along_axis(self.label, axis=1, arr=arr)
        return self.class_labels
        
    # Plots the synthetic data along with the class they belong to based on color
    def spread_plotter(self,p):
        class_flag = self.label(p)
        if len(p) !=0:
            if class_flag:
                plt.plot(p[0],p[1],'*g')
            else:
                plt.plot(p[0],p[1],'*b')

    # Visualize the complete dataset and also the lines used for the non-linear artifical class assignment, Only use this method for dimension = 2
    def visualize(self,arr):
        for i,plane in enumerate(self.planes):
            self.X = np.linspace(0,1,100)
            #print(self.A[0],self.A[1],self.B)
            self.Y = (plane[0][0]*self.X+plane[1])/(-plane[0][1])
            #print(self.X,self.Y)
            #plt.plot(self.X,self.Y)
            if i==0:
                plt.ion()
        #plt.plot(self.intersection[0],self.intersection[1],'*r')
        np.apply_along_axis(self.spread_plotter, axis=1, arr=arr)
        #np.apply_along_axis(self.spread_plotter, axis=1, arr=zip(arr,self.labeler(arr)))
        # Set the limits of x and y to be between 0 and 1
        plt.xlim([0, 1])
        plt.ylim([0, 1])

    # Visualize the complete dataset and choquet classification and also the lines used for the non-linear artifical class assignment, Only use this method for dimension = 2
    def visualize_choquet(self,arr,chromosome =None):
        if str(type(chromosome)) == "<class 'NoneType'>":
            self.chromosome = self.generate_chromosome()
        else:
            self.chromosome = chromosome
        print(self.chromosome)
        for i,plane in enumerate(self.planes):
            self.X = np.linspace(0,1,100)
            #print(self.A[0],self.A[1],self.B)
            self.Y = (plane[0][0]*self.X+plane[1])/(-plane[0][1])
            #print(self.X,self.Y)
            #plt.plot(self.X,self.Y)
            if i==0:
                plt.ion()
        #plt.plot(self.intersection[0],self.intersection[1],'*r')
        np.apply_along_axis(self.spread_plotter_choquet, axis=1, arr=arr)
        #np.apply_along_axis(self.spread_plotter, axis=1, arr=zip(arr,self.labeler(arr)))
        # Set the limits of x and y to be between 0 and 1
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
    # Generates gene
    def generate_gene(self, precision = 0.001):
        gene_size = int(np.ceil(np.log2(1/precision)))
        genes_bit_array = np.random.choice([0, 1], size=gene_size) 
        return ''.join(map(str, genes_bit_array))
    
    # Generates chromosome
    def generate_chromosome(self, dimensions = 2, precision = 0.001):
        # We consider a gene that represents a real number in the range [0, 1]. 
        # If we want to represent this number with a precision of 10^−3, 
        # we need to be able to distinguish 10^3=1000 different values within this range.
        # Therefore, log2(range/precision) = log2(1000)≈10 bits.
        genes_array = [self.generate_gene(precision) for i in range(int((np.power(2, dimensions))+(2*dimensions)))]
        chromosome = ''.join(map(str, genes_array))
        return chromosome
    
    # Decode a gene (convert from binary to decimal)
    def decode_gene(self, gene):
        return 2*((int(gene, 2)/(2**10-1))-0.5)
    
    # For inference of chromosome
    def infer_chromosome(self, chromosome, dimensions = 2, precision = 0.001):
        gene_size = int(np.ceil(np.log2(1/precision)))
        genes = [self.decode_gene(chromosome[i:i+gene_size]) for i in range(0, len(chromosome), gene_size)]
        mu,B,a,b = (genes[0:int(np.power(2,dimensions)-1)],genes[int(np.power(2,dimensions))-1],genes[int(np.power(2,dimensions)):int(np.power(2,dimensions)+dimensions)],genes[int(np.power(2,dimensions)+dimensions):int(np.power(2,dimensions)+2*dimensions)])
        mu = np.array(mu)
        a = np.array(a)
        b = np.array(b)
        return mu,B,a,b
    
    # Generalized Choquet integral with signed fuzzy measures
    def generalized_choquet_integral(self, p, mu, B, a, b, dimensions = 2):
        integral = 0.0   
        powerset = [[k for k in range(1,dimensions+1) if (((i/np.power(2,k)) - np.floor((i/np.power(2,k)))) < 1 and ((i/np.power(2,k)) - np.floor((i/np.power(2,k)))) >= 0.5)] for i in range(1,np.power(2,dimensions))]
        f = sorted(p)
        for i in range(dimensions):
            index = i+1
            if index != dimensions:
                mu_i_index = powerset.index([j for j in range(index,dimensions+1)]) 
                mu_i_1_index = powerset.index([j for j in range(index+1,dimensions+1)]) 
                integral = integral + ((a[index-1]+(b[index-1]*f[index-1]))*(mu[mu_i_index] - mu[mu_i_1_index]))
            else:
                mu_i_index = powerset.index([j for j in range(index,dimensions+1)])
                integral = integral + ((a[index-1]+(b[index-1]*f[index-1]))*(mu[mu_i_index]))
        return integral
    
    # Create genetic pool
    def create_population(self, population = 100, dimensions = 2, precision = 0.001):
        return [self.generate_chromosome(dimensions, precision) for i in range(population)]
    
    # Calculation of Choquet distance
    def choquet_distance(self, arr, class_labels, chromosome, dimensions = 2, precision = 0.001):
        distance = 0.0
        numerator = 0.0
        mu,B,a,b = self.infer_chromosome(chromosome, dimensions, precision)
        l = sum(class_labels)
        l_prime = len(class_labels)-l
        penalty = np.abs(l-l_prime)+1
        for i,p in enumerate(arr):
            choquet_integ = self.generalized_choquet_integral(p, mu, B, a, b, dimensions)
            c = 1.0
            if class_labels[i]:
                if choquet_integ < B:
                    c = penalty
                d_j =  choquet_integ - B
                numerator = numerator + (c*d_j)
            else:
                if choquet_integ > B:
                    c = penalty
                d_j =  B - choquet_integ
                numerator = numerator + (c*d_j)
        distance = numerator/np.sqrt(np.sum(mu**2))
        return distance
    
    # Calculate fitness
    def chromosome_fitness(self, distances):
        D = np.array(distances)
        min_D, max_D = np.min(D), np.max(D)
        if max_D == min_D:
            return np.zeros_like(D)  # or some other appropriate value
        else:
            return (D - min_D) / (max_D - min_D)
        #return (D-np.min(D))/(np.max(D) - np.min(D))
    
    # Genetic crossover
    def crossover(self, parent1, parent2):
        # Choose a random crossover point
        idx = np.random.randint(1, len(parent1)-1)
        # Create the child by taking all genes up to idx from parent 1 and all genes after idx from parent 2
        child1 = parent1[:idx] + parent2[idx:]
        child2 = parent2[:idx] + parent1[idx:]
        return child1,child2
    
    # Genetic mutation introduction
    def mutate(self, chromosome, mutation_rate=0.01):
        # Convert the chromosome string to a list of integers
        chromosome = list(map(int, chromosome))
        # Go through each gene in the chromosome
        for i in range(len(chromosome)):
            # With a probability of the mutation_rate, flip the gene
            if np.random.rand() < mutation_rate:
                chromosome[i] = sum([not chromosome[i]])
        # Convert the chromosome back to a string
        chromosome = ''.join(map(str, chromosome))
        return chromosome
    
    # Parent selection based on fitness and random switch
    def select_parents(self, people, fitnesses):
        # Check for zero sum of fitnesses
        if fitnesses.sum() == 0:
            # Add a small constant to fitnesses to avoid zero sum
            fitnesses += 1e-9
        # Check for NaN values in fitnesses
        if np.isnan(fitnesses).any():
            raise ValueError("Fitnesses contains NaN values")
        # Check for zero sum of fitnesses
        if fitnesses.sum() == 0:
            raise ValueError("Sum of fitnesses is zero")
        # Normalize the fitnesses to sum to 1, so they can be used as probabilities
        probabilities = fitnesses / fitnesses.sum()
        # Use np.random.choice to select two parents, with replacement
        parents = np.random.choice(people, size=2, replace=True, p=probabilities)
        return parents
    
    def create_new_generation(self, people, fitnesses, num_offspring, random_switch_prob = 0.5):
        new_generation = []
        for _ in range(num_offspring):
            # Select two parents
            parents = self.select_parents(people, fitnesses)
            # With a probability of a, perform crossover, otherwise perform mutation
            if np.random.rand() < random_switch_prob:
                child1,child2 = self.crossover(parents[0], parents[1])
            else:
                child1 = self.mutate(parents[0])
                child2 = self.mutate(parents[1])
            new_generation.append(child1)
            new_generation.append(child2)
        return new_generation
    
    # Genetic algorithm
    def learn(self, population=100, arr = None, class_labels = None, planes = None, dimensions = 2, precision = 0.001):
        if str(type(arr)) == "<class 'NoneType'>":
            arr = self.arr
        if str(type(class_labels)) == "<class 'NoneType'>":
            class_labels = self.class_labels
        if str(type(planes)) == "<class 'NoneType'>":
            planes = self.planes
        best_chromosome = ""
        generation_count = 0
        consecutive_low_count = 0
        people = self.create_population(population, dimensions, precision)
        D_max = 0.0
        D_min = 0.0
        pre_D = 0.0
        cur_D = 1.0
        print("generation_count","D_max","D_min","best_chromosome")
        while True:
            fitnesses = self.chromosome_fitness([self.choquet_distance(arr, class_labels, chromosome, dimensions, precision) for chromosome in people])
            new_generation = self.create_new_generation(people,fitnesses,int(population/2))
            people = people + new_generation
            distances = [self.choquet_distance(arr, class_labels, chromosome, dimensions, precision) for chromosome in people]
            fitnesses = self.chromosome_fitness(distances)
            sorted_indices = np.argsort(fitnesses)
            cur_D = distances[sorted_indices[-1]]
            D_max = np.max(distances)
            D_min = np.min(distances)
            best_chromosome = people[sorted_indices[-1]]
            if generation_count > 60:
                print("Exiting due to generation limit")
                break
            if np.abs(cur_D - pre_D) < 0.0001:
                consecutive_low_count = consecutive_low_count+1
                if generation_count > 30:
                    if consecutive_low_count > 10:
                        break
            else:
                consecutive_low_count = 0
            print(generation_count,D_max,D_min,best_chromosome)
            pre_D = cur_D
            people = [people[i] for i in sorted_indices[-population:]]
            generation_count = generation_count+1
        return best_chromosome,cur_D,generation_count
        
            
        

if __name__ == '__main__':
    initialize_plane = [(np.array([0.88478958, 0.33991662]), -0.3469162247673856),(np.array([0.40094829, 0.55576035]), -0.3580698348895637)]
    # Target Choquet hyper plane is mentioned as a chromosome "00100010000011010100111000001110110011011001100011011101100010111000101000111100"
    dg = SynthNonLinearClassifiedDataGen(dimensions=2,size=200, intersection=[0.2,0.5],intersection_flag = False,planes=initialize_plane,chromosome = "00100010000011010100111000001110110011011001100011011101100010111000101000111100")
    data = dg.generate_data()
    class_labels = dg.labeler_choquet(data)
    # planes = dg.get_hyper_planes()
    # print(planes)
    # dg.visualize(data)

    # Target classification
    # dg.visualize_choquet(data,"00100010000011010100111000001110110011011001100011011101100010111000101000111100")

    #class_labels = dg.labeler(data)
    plt.ioff()
    plt.show()
    # uncomment below to run the genetic algorithm
    #print(dg.learn())
    # Sample chromosome generated by running the algorithm
    # Use the chromsome generated using the convergent value from learn
    # dg.visualize_choquet(data,"01111111111010001011010011111100101000001111111001000011101100000000001010000000")