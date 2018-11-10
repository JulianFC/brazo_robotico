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
    QT, P = direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, obj):
        diff = diff + abs(coord - obj_coord)
    return 1 / (1+diff),


objectives = []
results = []
i = 0
fitness_evolution = []
X = []
while i < 10:
    i = i+1



    objective = [random.uniform(-2,2),random.uniform(-2,2),random.uniform(0,3)]

    toolbox.register("evaluate", evalOneMax, obj=objective)
    toolbox.register("mate", tools.cxUniform, indpb=0.25)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)

    NGEN=200
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        top = tools.selBest(population, k=1)[0]
        if (i == 9):
            fitness_evolution.append(evalOneMax(top, objective))
            X.append(gen)

    #top10 = tools.selBest(population, k=10)

    #animate([random.uniform(-pi, pi), random.uniform(-pi, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)], top)
    #print(top10)
    QT, P = direct_problem(top, 4)
    objectives.append(objective)
    results.append(P)
    #print(P)
    #print(objective)

print("Objetivo:\t\t\t\t\t\tResultado:\t\t\t\t\t\tError:\n")
for i in range(10):
    objectives[i] = [round(objectives[i][j], 4) for j in range(3)]
    results[i] = [round(results[i][j], 4) for j in range(3)]
    error = [round(abs(results[i][j]-objectives[i][j]), 4) for j in range(3)]
    print(str(objectives[i])+"\t"+str(results[i])+"\t"+str(error)+"\n")

plt.plot(X,fitness_evolution)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del fitness en el tiempo")
plt.show()

print(fitness_evolution)