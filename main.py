import random
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def init2d(icls, shape):
    return icls(np.random.uniform(-pi,pi, shape))


#toolbox.register("attr_bool", random.uniform, -pi, pi)
toolbox.register("individual", init2d, creator.Individual,  shape=(4,20))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual, obj):
    QT, P = direct_problem(individual, 4)
    diff = 0
    for (coord, obj_coord) in zip(P, obj):
        diff = diff + abs(coord - obj_coord)
    return 1 / (1+diff),

def energy_fitness(individual, start_p, final_p):
    energy = 0
    W = [1,3,2,1]

    print(individual.size())
    print("\n")
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

    return energy


objectives = []
results = []
i = 0
fitness_evolution = []
X = []
while i < 1:
    i = i+1

    start = [random.uniform(-pi,pi), random.uniform(-pi,pi), random.uniform(-pi,pi),random.uniform(-pi,pi)]
    objective = [random.uniform(-pi,pi),random.uniform(-pi,pi),random.uniform(-pi,pi),random.uniform(-pi,pi)]

    toolbox.register("evaluate", energy_fitness, start_p = start, final_p = objective)
    toolbox.register("mate", tools.cxUniform, indpb=0.25)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=200)

    NGEN=100
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        top = tools.selBest(population, k=1)[0]
        if (i == 1):
            fitness_evolution.append(energy_fitness(top, start, objective))
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
