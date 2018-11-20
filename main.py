import random
import numpy as np
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def init2d(icls, shape, start, objective):
    tray0 = np.linspace(start[0],objective[0],20)
    tray1 = np.linspace(start[1], objective[1], 20)
    tray2 = np.linspace(start[2], objective[2], 20)
    tray3 = np.linspace(start[3], objective[3], 20)


    Theta = []
    for i in range(20):
        Theta.append(np.array([tray0[i],tray1[i],tray2[i],tray3[i]]))
    Theta = Theta + np.random.uniform(-0.75,0.75,shape)
    return icls(Theta)

start = [random.uniform(-pi,pi), random.uniform(-pi,pi), random.uniform(-pi,pi),random.uniform(-pi,pi)]
objective = [random.uniform(-pi,pi),random.uniform(-pi,pi),random.uniform(-pi,pi),random.uniform(-pi,pi)]

#toolbox.register("attr_bool", random.uniform, -pi, pi)
toolbox.register("individual", init2d, creator.Individual,  shape=(20,4), start = start, objective = objective)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def crossover(ind1, ind2, indpb):
    for i in range(4):
        if random.random() < indpb:
            lam = random.random()
            ind1[i], ind2[i] = lam*ind1[i]+(1-lam)*ind2[i], lam*ind2[i]+(1-lam)*ind1[i]
    return ind1, ind2

def evalOneMax(individual, obj):
    QT, P = direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, obj):
        diff = diff + abs(coord - obj_coord)
    return 1 / (1+diff),

def energy_fitness(individual, start_p, final_p):
    energy = 0
    W = [1.5,3,2,1]

    for i in range(21):
        delta = 0
        for j in range(4):
            if i == 0:
                delta = delta + W[j]*abs(start_p[j] - individual[i][j])
            elif i == 20:
                delta = delta + W[j]*abs(final_p[j] - individual[i-1][j])
            else:
                delta = delta + W[j] * abs(individual[i][j] - individual[i - 1][j])
        energy = energy + delta

    return energy,


objectives = []
results = []
i = 0
fitness_evolution = []
X = []
while i < 1:
    i = i+1

    toolbox.register("evaluate", energy_fitness, start_p = start, final_p = objective)
    toolbox.register("mate", crossover, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.75, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=10)

    population = toolbox.population(n=100)

    NGEN=300
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.6)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        top = tools.selBest(population, k=1)[0]
        if (i == 1):
            fitness_evolution.append(energy_fitness(top, start, objective))
            X.append(gen)

    #top10 = tools.selBest(population, k=10)

    #
    #print(top10)
    QT = []
    P = []
    for i in range(20):
        qt, p = direct_problem(top[i], 4)
        #print(top[i])
        QT.append(qt)
        P.append(p)
    objectives.append(objective)
    results.append(P)

Theta0 = []
Theta1 = []
for i in range(4):
    print("[", end='')
    for j in range(20):
        if i == 0:
            Theta0.append(top[j][i])
        if i == 1:
            Theta1.append(top[j][i])
        print(str(top[j][i])+", ",end = '')
    print("]")
#animate_path(top)
#animate(top[0], top[19])
    #print(P)
    #print(objective)

# print("Objetivo:\t\t\t\t\t\tResultado:\t\t\t\t\t\tError:\n")
# for i in range(10):
#     objectives[i] = [round(objectives[i][j], 4) for j in range(3)]
#     results[i] = [round(results[i][j], 4) for j in range(3)]
#     error = [round(abs(results[i][j]-objectives[i][j]), 4) for j in range(3)]
#     print(str(objectives[i])+"\t"+str(results[i])+"\t"+str(error)+"\n")
#
plt.plot(X,fitness_evolution)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del fitness en el tiempo")
plt.show()
plt.plot(Theta0)
plt.plot(Theta1)
plt.legend(["Angulo 1", "Angulo 2"])
plt.show()
print(fitness_evolution[-1])
#
# print(fitness_evolution)