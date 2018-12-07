import random
import numpy as np
from deap import creator, base, tools, algorithms
from math import pi
from denavit_hartenberg import direct_problem
from operator import attrgetter
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

def mutGaussian(individual, mu, sigma, indpb,alpha):
    M = len(individual)
    if random.random() <= alpha:
        for i in range(M):
            for j in range(4):
                if random.random() < indpb:
                    individual[i][j] += random.gauss(mu, sigma)
    else:
        sigma = np.random.uniform(M/7,M/2,4)
        mu = np.random.uniform(-M/8,M/8,4)+M/2
        x = np.linspace(0,M-1,M)
        h = np.random.uniform(-1,1,4)
        for j in range(4):
            gauss = np.exp(-np.power(x - mu[j], 2.) / (2 * np.power(sigma[j], 2.)))*h[j]
            for i in range(M):
                individual[i][j] += gauss[i]

    return individual,

def selection(individuals, k, tournsize, alpha, fit_attr="fitness"):
    chosen = []
    N = int(k*alpha)
    for i in range(N):
        aspirants = tools.selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))

    new = tools.selRandom(individuals,k-N)
    return chosen+new

def avg_fitness(population,sp,fp):
    fitness = np.array([])
    for ind in population:
        fitness = np.append(fitness, energy_fitness(ind,sp,fp))
    return np.average(fitness)


def get_path(start_p, final_p, M, plot=False, matlab=True):
    toolbox = base.Toolbox()
    toolbox.register("individual", init2d, creator.Individual, shape=(M, 4), start=start_p, objective=final_p)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", energy_fitness, start_p=start_p, final_p=final_p)
    toolbox.register("mate", crossover, indpb=0.45)
    toolbox.register("mutate", mutGaussian, mu=0, sigma=0.05, indpb=0.3, alpha=0.6)
    toolbox.register("select", selection, tournsize=3, alpha=0.8)

    population = toolbox.population(n=150)

    objectives = []
    results = []
    max_fitness_evolution = []
    avg_fitness_evolution = []
    X = []

    NGEN = 1000
    gen0 = 0
    gen1 = int(NGEN / 3)
    gen2 = int(NGEN * 2 / 3)
    for gen in range(NGEN):
        avg_fitness_evolution.append(avg_fitness(population, start_p, final_p))
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        if gen == gen0:
            top0 = tools.selBest(population, k=1)[0]

        #if gen == 5:# ARREGLAR CROSSOVER
        ##   best0 = tools.selBest(population, k=10)
         #  for i in range(10):
         #     print(best0[i])
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

        plt.plot(X, max_fitness_evolution)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Evolución del mejor fitness y el fitness promedio en el tiempo")
        plt.plot(X, avg_fitness_evolution)
        plt.legend(["Mejor", "Promedio"])
        plt.show()

    if matlab:
        # Matlab animation:
        Theta0 = []
        Theta1 = []
        Theta2 = []
        Theta3 = []
        for i in range(4):
            print("[", end='')
            for j in range(M):
                if i == 0:
                    Theta0.append(top0[j][i])
                if i == 1:
                    Theta1.append(top0[j][i])
                if i == 2:
                    Theta2.append(top0[j][i])
                if i == 3:
                    Theta3.append(top0[j][i])
                if j == M - 1:
                    print(str(top0[j][i]), end='')
                else:
                    print(str(top0[j][i]) + ", ", end='')
            print("];")
        print("\n")
        Theta0 = []
        Theta1 = []
        Theta2 = []
        Theta3 = []
        for i in range(4):
            print("[", end='')
            for j in range(M):
                if i == 0:
                    Theta0.append(top1[j][i])
                if i == 1:
                    Theta1.append(top1[j][i])
                if i == 2:
                    Theta2.append(top1[j][i])
                if i == 3:
                    Theta3.append(top1[j][i])
                if j == M - 1:
                    print(str(top1[j][i]), end='')
                else:
                    print(str(top1[j][i]) + ", ", end='')
            print("];")
        Theta0 = []
        Theta1 = []
        Theta2 = []
        Theta3 = []
        print("\n")
        for i in range(4):
            print("[", end='')
            for j in range(M):
                if i == 0:
                    Theta0.append(top2[j][i])
                if i == 1:
                    Theta1.append(top2[j][i])
                if i == 2:
                    Theta2.append(top2[j][i])
                if i == 3:
                    Theta3.append(top2[j][i])
                if j == M - 1:
                    print(str(top2[j][i]), end='')
                else:
                    print(str(top2[j][i]) + ", ", end='')
            print("];")
        Theta0 = []
        Theta1 = []
        Theta2 = []
        Theta3 = []
        print("\n")
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
                if j == M - 1:
                    print(str(topF[j][i]), end='')
                else:
                    print(str(topF[j][i]) + ", ", end='')
            print("];")



    print("\nUltimo fitness: " + str(max_fitness_evolution[-1]))
    # Python animation:
    #animate(start_p,topF[-1])
    animate_path(top0, "Top0.gif")
    animate_path(top1, "Top1.gif")
    animate_path(top2, "Top2.gif")
    animate_path(topF, "Top3.gif")
    return

