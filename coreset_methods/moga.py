import random, multiprocessing, os, sys
import numpy as np
import matplotlib.pyplot as plt

from math import dist, e, log
from deap import base, creator, tools, algorithms
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
from visualizer import plot_fitness_evolution, plot_fitness_scatter, create_video

# sys.setrecursionlimit(1)
class MOGA(BaseEstimator, TransformerMixin):
    def __init__(self, x, y, classifier, metric, solutionSize = 1, sizePop = 100, numGen = 100, cxPb = 0.8, indMPb = 1, mutPb = 0.07, 
                crowding = 0.3, coreset = True, size = -1.0, distance = False, classDist = None, seed = 1, name = "naive"):
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
        self.num_classes = len(np.unique(y))
        self.classifier = classifier
        self.metric = metric
        
        self.sizePop = sizePop
        self.numGen = numGen
        self.cxPb = cxPb  # Crossover probability
        self.indMPb = indMPb  # Individual mutation probability
        self.mutPb = mutPb  # Mutation probability
        self.crowding = crowding  # Crowding factor for NSGA-II
        self.coreset = coreset  # Whether to use a coreset
        self.size = size  # Size of the coreset, if applicable
        self.distance = distance  # Whether to consider distance in fitness
        self.classDist = classDist  # Class distribution, if applicable
        self.seed = seed
        self.purity = entropy(y)
        self.fitness_list = []
        self.name = name
        self.solutionSize = solutionSize

        if coreset:
            if not (size == 1.0 or size == -1.0): # size fixed
                if distance and (classDist is not None):
                    creator.create("Fitness", base.Fitness, weights = (1.0, 1.0, -1.0,)) #(max performance, max distance +/-, min classDist(min difference of ratio or baseline))
                elif distance:
                    creator.create("Fitness", base.Fitness, weights = (1.0, 1.0,)) #(max performance, max distance +/-)
                elif classDist is not None:
                    creator.create("Fitness", base.Fitness, weights = (1.0, -1.0,)) #(max performance, min classDist(min difference of ratio or baseline))
                else:
                    creator.create("Fitness", base.Fitness, weights = (1.0,)) #(max performance)
            else: # size not fixed
                if distance and (classDist is not None):
                    creator.create("Fitness", base.Fitness, weights = (1.0, size, 1.0, -1.0,)) #(max performance,max/min size(-1 by default), max distance +/-, min classDist(min difference of ratio or baseline))
                elif distance:
                    creator.create("Fitness", base.Fitness, weights = (1.0, size, 1.0,)) #(max performance, max/min size(-1 by default), max distance +/-)
                elif classDist is not None:
                    creator.create("Fitness", base.Fitness, weights = (1.0, size, -1.0,)) #(max performance, max/min size(-1 by default), min classDist(min difference of ratio or baseline))
                else:
                    creator.create("Fitness", base.Fitness, weights = (1.0, size)) #(max performance, max/min size(-1 by default))
                
                # print(f"============== Fitness initialization: {creator.Fitness.weights} ===============")
                creator.create("Individual", list, fitness=creator.Fitness)
                self.toolbox = base.Toolbox()
                ref_points = tools.uniform_reference_points(len(creator.Fitness.weights), sizePop)
                # self.toolbox.register("attr_bool", random.randint, 0, 1)
                # self.toolbox.register("attr_bool", lambda: 1)
                # self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, len(self.x))
                # self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
                
                def init_individual(creator, size, num_classes):
                    """Initializes an individual with a random number of 1s between `num_classes` and `size`, rest 0s."""
                    num_coreset = random.randint(num_classes, size)  # Randomly determine the number of 1s
                    individual = [0] * size  # Start with all 0s
                    coreset_indices = random.sample(range(size), num_coreset)  # Randomly pick indices for 1s
                    for index in coreset_indices:
                        individual[index] = 1  # Set chosen indices to 1
                    return creator.Individual(individual)  # Ensure it returns an instance of creator.Individual

                # Assuming 'self' is an object that contains the toolbox, and 'x' and 'num_classes' are defined
                self.toolbox.register("individual_custom", init_individual, creator, size=len(self.X_train), num_classes=len(np.unique(self.y_train)))  # num_classes needs to be defined
                self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.individual_custom)
                self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
                
                self.toolbox.register("evaluate", fitness_evaluation, classifier=classifier, metric=get_scorer(metric), X_train=self.X_train, 
                                        X_validate=self.X_validate, y_train=self.y_train, y_validate=self.y_validate, size=size, coreset=coreset, distance=distance,
                                        classDist=classDist, purity=self.purity, seed=seed, num_classes = self.num_classes)
                self.toolbox.register('mate', tools.cxOnePoint)
                self.toolbox.register('mutate', tools.mutFlipBit, indpb = mutPb)
                # self.toolbox.register("NSGAselect", tools.selNSGA3, ref_points=ref_points)
                self.toolbox.register("NSGAselect", tools.selNSGA2)
        else:
            # for creating validation set
            pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        '''
        Evolution starts
        '''
        pop = self.toolbox.population(n=self.sizePop)
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(1, self.numGen + 1):
            print(f"\n-- Generation {gen} --")

            offspring = list(map(self.toolbox.clone, pop))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxPb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.indMPb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop = self.toolbox.NSGAselect(pop + offspring, self.sizePop)

            # Gather all the fitnesses in one list and calculate stats for each objective
            fits = np.array([ind.fitness.values for ind in pop])
            # plot_fitness_scatter(fits, gen, "MOGA")
            gen_stats = []  # Store max, min, and avg for each objective in this generation
            for i in range(fits.shape[1]):
                max_fit = np.max(fits[:, i])
                min_fit = np.min(fits[:, i])
                avg_fit = np.mean(fits[:, i])
                gen_stats.append((max_fit, min_fit, avg_fit))
                print(f" F{i+1}: Avg = {round(avg_fit,3)}, Max = {round(max_fit, 3)}, Min = {round(min_fit,3)}")

            self.fitness_list.append(gen_stats)  # Append the stats for this generation

        if self.distance and (self.classDist is not None):
            pass
        elif self.distance:
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[2], reverse=True)
        elif self.classDist is not None:
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[2], reverse=False)
        else:
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
            
        optimal_solutions = sorted_pop[:self.solutionSize]
        indices_list = [convert_indices(ind, self.size) for ind in optimal_solutions]
        self.optimal_solutions = pop
        # plot_fitness_evolution(self.fitness_list, self.name)
        return [self.X_train[indices] for indices in indices_list], [self.y_train[indices] for indices in indices_list], indices_list


# # wrapper function
# def evaluate_individual(args):
#     ind, classifier, metric, x, y, size, coreset, distance, classDist, seed = args
#     return fitness_evaluation(ind, classifier, metric, x, y, size, coreset, distance, classDist, seed)

def fitness_evaluation(ind, classifier, metric,  X_train, X_validate, y_train, y_validate, size, coreset, distance, classDist, purity, seed, num_classes):
    indices = convert_indices(ind, size)
    core_x = X_train[indices]
    core_y = y_train[indices]
    if coreset:
            if not (size == 1.0 or size == -1.0): # size fixed
                if distance and (classDist is not None):
                    creator.create("Fitness", base.Fitness, weights = (1.0, 1.0, -1.0,)) #(max performance, max distance +/-, min classDist(min difference of ratio or baseline))
                elif distance:
                    creator.create("Fitness", base.Fitness, weights = (1.0, 1.0,)) #(max performance, max distance +/-)
                elif classDist is not None:
                    creator.create("Fitness", base.Fitness, weights = (1.0, -1.0,)) #(max performance, min classDist(min difference of ratio or baseline))
                else:
                    creator.create("Fitness", base.Fitness, weights = (1.0,)) #(max performance)
            else: # size not fixed
                if distance and (classDist is not None):
                    return (perform_train_validation(core_x, core_y, X_validate, y_validate, classifier, metric, seed, num_classes), len(indices), distance_class(core_x, core_y), manipulate_distribution(core_y, y_train, purity, classDist),)
                elif distance:
                    return (perform_train_validation(core_x, core_y, X_validate, y_validate, classifier, metric, seed, num_classes), len(indices), distance_class(core_x, core_y),)
                elif classDist is not None:
                    return (perform_train_validation(core_x, core_y, X_validate, y_validate, classifier, metric, seed, num_classes), len(indices), manipulate_distribution(core_y, y_train, purity, classDist),)
                else:
                    return (perform_train_validation(core_x, core_y, X_validate, y_validate, classifier, metric, seed, num_classes), len(indices),) #(max performance, max/min size(-1 by default))
    else:
        pass

def perform_train_validation(X, y, X_validate, y_validate, classifier, metric, seed, num_classes):
    if not len(np.unique(y)) == num_classes:
        return 0
    classifier.fit(X, y)
    return metric(classifier, X_validate, y_validate)

# def balance_data_fitness(y):
#     counts = np.bincount(y)
#     baseline = 1.0 / len(counts)
#     minority_count = counts.min()  
#     minority_ratio = minority_count / len(y)
#     difference = baseline - minority_ratio
#     return difference # minimize this

def convert_indices(ind, size):
    if not (size == 1.0 or size == -1.0):
        pass
    else:
        indices = [i for i, bit in enumerate(ind) if bit == 1]
        return indices

# def distance_class(x, y):
#     classes = np.unique(y)
#     centers = np.array([x[np.where(y == clas)].mean(axis=0) for clas in classes])
#     # Compute distances between each pair of class centers
#     distances = []
#     for i in range(len(centers)):
#         for j in range(i + 1, len(centers)):
#             distance = dist(centers[i].flatten(), centers[j].flatten())
#             print(distance)
#             distances.append(distance)
#     # Calculate the average distance
#     avg_distance = np.mean(distances)
#     # if avg_distance == NaN:
#     #     return 0
#     return avg_distance

def distance_class(x, y):
    classes = np.unique(y)
    centers = []
    for clas in classes:
        class_samples = x[np.where(y == clas)]
        center = class_samples.mean(axis=0)
        centers.append(center)

    # Compute distances between each pair of class centers
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance = dist(centers[i].flatten(), centers[j].flatten())
            distances.append(distance)

    # Check if distances list is empty before computing mean if there is only one class
    if distances:
        avg_distance = np.mean(distances)
    else:
        avg_distance = 0 

    return avg_distance

def manipulate_distribution(core_y, y_train, purity, classDist):
    if classDist == "preserve":
        return compute_kl_divergence(y_train, core_y)
    elif classDist == "balance":
        return entropy(core_y)

def gini_index(labels):
    counts = np.bincount(labels.flatten())  # Fast counting of occurrences for non-negative integers
    total_instances = counts.sum()
    if total_instances == 0:
        return 0.0  # To avoid division by zero
    
    proportion_squared = (counts / total_instances) ** 2
    gini = 1.0 - proportion_squared.sum()
    return gini

def entropy(labels, base=None):
    n_labels = len(labels)
    if n_labels <= 1: return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1: return 0
    ent = 0.
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return -ent #here I return the negative value of entropy to minimize, which equals maximizing the positive entropy

def compute_kl_divergence(y, core_y):

    def compute_class_probabilities(labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()

        return dict(zip(unique_labels, probabilities))

    original_probs = compute_class_probabilities(y)
    subset_probs = compute_class_probabilities(core_y)

    for clas in list(original_probs.keys()):
        if clas not in subset_probs:
            subset_probs[clas] = 0

    p = np.array([original_probs[clas] for clas in list(original_probs.keys())])
    q = np.array([subset_probs[clas] for clas in list(original_probs.keys())] + np.finfo(float).eps)  # Add epsilon to avoid division by zero

    return np.sum(p * np.log(p / q))


