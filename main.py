import random
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.uniform, -pi, pi)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual, obj):
    QT, P= direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, obj):
        diff = diff+abs(coord-obj_coord)
    return 1/diff,

objective = [random.uniform(-2,2),random.uniform(-2,2),random.uniform(0,3)]

toolbox.register("evaluate", evalOneMax, obj=objective)
toolbox.register("mate", tools.cxUniform, indpb=0.25)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN=100
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    #if gen%20 == 0:
    #    top = tools.selBest(population, k=1)[0]
    #    animate([0, 0, 0, 0], top)
top10 = tools.selBest(population, k=10)
top = tools.selBest(population, k=1)[0]
animate([random.uniform(-pi, pi), random.uniform(-pi, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)], top)
#print(top10)
QT, P = direct_problem(top, 4)
print(P)
print(objective)
