import random
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *



def fitness_point(individual, obj):

    QT, P = direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, obj):
        diff = diff + abs(coord - obj_coord)
    return diff,


def init(objective):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.uniform, -pi, pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.25)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness_point, obj=objective)
    return toolbox

def get_individual(toolbox):
    population = toolbox.population(n=150)
    NGEN = 100
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    top = tools.selBest(population, k=1)[0]

    return top
