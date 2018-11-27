import random
import point_function
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *

objective = [random.uniform(-2,2),random.uniform(-2,2),random.uniform(0,3)]
toolbox_point = point_function.init(objective)


#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_bool",  point_function.get_individual, toolbox_point)
#toolbox.register("individual", tools.initRepeat, , toolbox.attr_bool, n=4)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n = 1)
toolbox.register("population", tools.initRepeat, list, toolbox.attr_bool)


def energy_fitness(individual, start_p, objective):
    QT, P = direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, objective):
        diff = diff + abs(coord - obj_coord)
    if diff > 0.01:
        return 10000,

    energy = 0
    M = 4
    W = [1.5, 3, 2, 1]
    for i in range(M):
        energy += W[i]*abs(start_p[i] - individual[i])**2
    return energy,

start_p = [0,pi/4,-pi/4,-pi/4]
objectives = []
results = []
angles = []
i = 0
fitness_evolution = []
X = []


toolbox.register("evaluate", energy_fitness, objective=objective, start_p = start_p)
toolbox.register("mate", tools.cxUniform, indpb=0.25)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=75)
NGEN=100
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top = tools.selBest(population, k=1)[0]
    fitness_evolution.append(energy_fitness(top,start_p, objective))
    X.append(gen)
top10 = tools.selBest(population, k=10)

QT, P = direct_problem(top, 4)

plt.plot(X,fitness_evolution)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del fitness en el tiempo")
plt.show()
print("Objective: "+str(objective))
print("result: "+ str(P))
print(fitness_evolution)