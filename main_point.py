import random
import point_function
import path_function
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
        diff = diff + abs(coord - obj_coord)**2
    if diff > 0.1:
        return 10000,

    energy = 0
    M = 4
    W = [1.5, 3, 2, 1]
    for i in range(M):
        energy += W[i]*abs(start_p[i] - individual[i])
    return energy,

def avg_fitness(population,sp,fp):
    fitness = np.array([])
    for ind in population:
        fitness = np.append(fitness, energy_fitness(ind,sp,fp))
    return np.average(fitness)


avg_fitness_evolution = []
start_p = [0,pi/4,-pi/4,-pi/4]
objectives = []
results = []
angles = []
i = 0
fitness_evolution = []
X = []


toolbox.register("evaluate", energy_fitness, objective=objective, start_p = start_p)
toolbox.register("mate", tools.cxBlend, alpha=0)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
NGEN=150
gen0 = 0
gen1 = int(NGEN*1/3)
gen2 = int(NGEN*2/3)
gen3 = NGEN
for gen in range(NGEN):
    avg_fitness_evolution.append(avg_fitness(population, start_p, objective))
    offspring = algorithms.varAnd(population, toolbox, cxpb=0, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top = tools.selBest(population, k=1)[0]
    fitness_evolution.append(energy_fitness(top, start_p, objective))
    X.append(gen)

    if gen == gen0:
        top0 = top
    if gen == gen1:
        top1 = top
    if gen == gen2:
        top2 = top

    top3 = top

top10 = tools.selBest(population, k=10)

QT, P = direct_problem(top, 4)

plt.plot(X, fitness_evolution)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del mejor fitness y el fitness promedio en el tiempo")
plt.plot(X, avg_fitness_evolution)
plt.legend(["Mejor", "Promedio"])
plt.show()

plt.plot([1,2,3,4], start_p,"ro")
plt.plot([1,2,3,4], top0,"bo", markerfacecolor="None")
plt.plot([1,2,3,4], top1,"go", markerfacecolor="None")
plt.plot([1,2,3,4], top2,"yo", markerfacecolor="None")
plt.plot([1,2,3,4], top3,"k+")
plt.legend(["Punto inicial", "Generación "+str(gen0), "Generación "+str(gen1), "Generación "+str(gen2), "Generación "+str(gen3)])
plt.show()
print("Objective: " + str(objective))
print("result: " + str(P))
print(fitness_evolution)

best0 = path_function.get_path(start_p, top0, 20, plot=True)
best = path_function.get_path(start_p, top3, 20, plot=True)