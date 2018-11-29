import random
import numpy as np
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from arm_animation import *


def init2d(icls, shape, start, objective):
    M = shape[0]
    tray0 = np.linspace(start[0],objective[0],M)
    tray1 = np.linspace(start[1], objective[1], M)
    tray2 = np.linspace(start[2], objective[2], M)
    tray3 = np.linspace(start[3], objective[3], M)


    Theta = []
    for i in range(M):
        Theta.append(np.array([tray0[i],tray1[i],tray2[i],tray3[i]]))
    Theta = Theta + np.random.uniform(-0.75,0.75,shape)
    return icls(np.random.uniform(-pi,pi,shape))



def energy_fitness(individual, start_p, final_p):
    energy = 0
    M = len(individual)
    W = [1.5, 3, 2, 1]

    for i in range(M+1):
        delta = 0
        for j in range(4):
            if i == 0:
                delta = delta + W[j]*abs(start_p[j] - individual[i][j])**2
            elif i == M:
                delta = delta + W[j]*abs(final_p[j] - individual[i-1][j])**2
            else:
                delta = delta + W[j] * abs(individual[i][j] - individual[i - 1][j])**2
        energy = energy + delta
    return energy,


def crossover(ind1, ind2, indpb):
    for i in range(4):
        if random.random() < indpb:
            lam = random.random()
            ind1[i], ind2[i] = lam*ind1[i]+(1-lam)*ind2[i], lam*ind2[i]+(1-lam)*ind1[i]
    return ind1, ind2

def mutGaussian(individual, mu, sigma, indpb):
    M = len(individual)
    for i in range(M):
        for j in range(4):
            if random.random() < indpb:
                individual[i][j] += random.gauss(mu, sigma)
    return individual,

def avg_fitness(population,sp,fp):
    fitness = np.array([])
    for ind in population:
        fitness = np.append(fitness, energy_fitness(ind,sp,fp))
    return np.average(fitness)


def get_path(start_p, final_p, M, plot=False):
    toolbox = base.Toolbox()
    toolbox.register("individual", init2d, creator.Individual, shape=(M, 4), start=start_p, objective=final_p)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", energy_fitness, start_p=start_p, final_p=final_p)
    toolbox.register("mate", crossover, indpb=0.75)
    toolbox.register("mutate", mutGaussian, mu=0, sigma=0.05, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=4)

    population = toolbox.population(n=100)

    objectives = []
    results = []
    max_fitness_evolution = []
    avg_fitness_evolution = []
    X = []

    NGEN = 800
    gen0 = 0
    gen1 = int(NGEN / 3)
    gen2 = int(NGEN * 2 / 3)
    for gen in range(NGEN):
        avg_fitness_evolution.append(avg_fitness(population, start_p, final_p))
        offspring = algorithms.varAnd(population, toolbox, cxpb=1, mutpb=1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        if gen == gen0:
            top0 = tools.selBest(population, k=1)[0]

        # if gen == 5:# ARREGLAR CROSSOVER
        #    best0 = tools.selBest(population, k=10)
        #    for i in range(10):
        #       print(best0[i])
        if gen == gen1:
            top1 = tools.selBest(population, k=1)[0]

        if gen == gen2:
            top2 = tools.selBest(population, k=1)[0]
        topF = tools.selBest(population, k=1)[0]
        max_fitness_evolution.append(energy_fitness(topF, start_p, final_p)[0])
        X.append(gen)
    if plot:
        QT = []
        P = []
        for i in range(M):
            qt, p = direct_problem(topF[i], 4)

            QT.append(qt)
            P.append(p)
        objectives.append(final_p)
        results.append(P)

        Theta0 = [[], [], [], []]
        Theta1 = [[], [], [], []]
        Theta2 = [[], [], [], []]
        ThetaF = [[], [], [], []]
        for i in range(4):
            for j in range(M):
                Theta0[i].append(top0[j][i])
                Theta1[i].append(top1[j][i])
                Theta2[i].append(top2[j][i])
                ThetaF[i].append(topF[j][i])
            plt.plot(Theta0[i])
            plt.plot(Theta1[i])
            plt.plot(Theta2[i])
            plt.plot(ThetaF[i])
            plt.legend(["Generación " + str(gen0), "Generación " + str(gen1), "Generación " + str(gen2),
                        "Generación " + str(NGEN)])
            plt.title("Angulo " + str(i))
            plt.show()

        # Matlab animation:
        Theta0 = []
        Theta1 = []
        Theta2 = []
        Theta3 = []
        for i in range(4):
            print("[", end='')
            for j in range(M):
                if i == 0:
                    Theta0.append(topF[j][i])
                if i == 1:
                    Theta1.append(topF[j][i])
                if i == 2:
                    Theta2.append(topF[j][i])
                if i == 3:
                    Theta3.append(topF[j][i])
                print(str(topF[j][i]) + ", ", end='')
            print("]")

        plt.plot(X, max_fitness_evolution)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Evolución del mejor fitness y el fitness promedio en el tiempo")
        plt.plot(X, avg_fitness_evolution)
        plt.legend(["Mejor", "Promedio"])
        plt.show()

        print("Ultimo fitness: " + str(max_fitness_evolution[-1]))
        # Python animation:
        # animate_path(topF)
    return

