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
    return icls(np.random.uniform(-pi,pi,shape))

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

def mutGaussian(individual, mu, sigma, indpb):
    size = len(individual)
    for i in range(20):
        for j in range(4):
            if random.random() < indpb:
                individual[i][j] += random.gauss(mu, sigma)

    return individual,

objectives = []
results = []
i = 0
fitness_evolution = []
X = []


toolbox.register("evaluate", energy_fitness, start_p = start, final_p = objective)
toolbox.register("mate", crossover, indpb=0.25)
toolbox.register("mutate", mutGaussian, mu=0, sigma=0.5, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

population = toolbox.population(n=100)
NGEN=1000
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=1, mutpb=1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    if gen == 0:
        top0 = tools.selBest(population, k=1)[0]
    if gen == 100:
        top100 = tools.selBest(population, k=1)[0]
    if gen == 200:
        top200 = tools.selBest(population, k=1)[0]
    top300 = tools.selBest(population, k=1)[0]
    fitness_evolution.append(energy_fitness(top300, start, objective))
    X.append(gen)

#top10 = tools.selBest(population, k=10)

#
#print(top10)
QT = []
P = []
for i in range(20):
    qt, p = direct_problem(top300[i], 4)
    #print(top[i])
    QT.append(qt)
    P.append(p)
objectives.append(objective)
results.append(P)

Theta0 = []
Theta1 = []
for i in range(4):
    for j in range(20):
        if i == 0:
            Theta0.append(top0[j][i])
        if i == 1:
            Theta1.append(top0[j][i])
plt.plot(Theta0)
plt.plot(Theta1)
plt.legend(["Angulo 1", "Angulo 2"])
plt.title("Top gen 0")
plt.show()

Theta0 = []
Theta1 = []
for i in range(4):
    for j in range(20):
        if i == 0:
            Theta0.append(top100[j][i])
        if i == 1:
            Theta1.append(top100[j][i])
plt.plot(Theta0)
plt.plot(Theta1)
plt.legend(["Angulo 1", "Angulo 2"])
plt.title("Top gen 100")
plt.show()

Theta0 = []
Theta1 = []
for i in range(4):
    for j in range(20):
        if i == 0:
            Theta0.append(top200[j][i])
        if i == 1:
            Theta1.append(top200[j][i])
plt.plot(Theta0)
plt.plot(Theta1)
plt.legend(["Angulo 1", "Angulo 2"])
plt.title("Top gen 200")
plt.show()

Theta0 = []
Theta1 = []
Theta2 = []
Theta3 = []
for i in range(4):
    print("[", end='')
    for j in range(20):
        if i == 0:
            Theta0.append(top300[j][i])
        if i == 1:
            Theta1.append(top300[j][i])
        if i == 2:
            Theta2.append(top300[j][i])
        if i == 3:
            Theta3.append(top300[j][i])
        print(str(top300[j][i])+", ", end='')
    print("]")
plt.plot(Theta0)
plt.plot(Theta1)
plt.plot(Theta2)
plt.plot(Theta3)
plt.legend(["Angulo 1", "Angulo 2", "Angulo 3", "Angulo 4"])
plt.title("Top last gen")
plt.show()

plt.plot(X, fitness_evolution)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del fitness en el tiempo")
plt.show()

print(fitness_evolution[-1])

animate_path(top300)
#animate(top300[0], top300[19])
    #print(P)
    #print(objective)

# print("Objetivo:\t\t\t\t\t\tResultado:\t\t\t\t\t\tError:\n")
# for i in range(10):
#     objectives[i] = [round(objectives[i][j], 4) for j in range(3)]
#     results[i] = [round(results[i][j], 4) for j in range(3)]
#     error = [round(abs(results[i][j]-objectives[i][j]), 4) for j in range(3)]
#     print(str(objectives[i])+"\t"+str(results[i])+"\t"+str(error)+"\n")
#

#plt.plot(Theta0)
#plt.plot(Theta1)
#plt.legend(["Angulo 1", "Angulo 2"])
#plt.show()

#
# print(fitness_evolution)
