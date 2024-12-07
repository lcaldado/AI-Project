#!/usr/bin/env python3
import numpy as np
import skfuzzy as skf
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from MFIS_Classes import *

# -----------GIVEN FUNCTIONS----------

def readFuzzySetsFile(fleName):
    """
    This function reads a file containing fuzzy set descriptions
    and returns a dictionary with all of them
    """
    fuzzySetsDict = FuzzySetsDict() # dictionary to be returned
    inputFile = open(fleName, 'r')
    line = inputFile.readline()
    while line != '':
        fuzzySet = FuzzySet()   # just one fuzzy set
        elementsList = line.split(', ')
        setid = elementsList[0]
        var_label=setid.split('=')
        fuzzySet.var=var_label[0]
        fuzzySet.label=var_label[1]

        xmin = int(elementsList[1])
        xmax = int(elementsList[2])
        a = int(elementsList[3])
        b = int(elementsList[4])
        c = int(elementsList[5])
        d = int(elementsList[6])
        x = np.arange(xmin-1,xmax+1)
        y = skf.trapmf(x, [a, b, c, d])
        fuzzySet.x = x
        fuzzySet.y = y
        fuzzySetsDict.update( { setid : fuzzySet } )

        line = inputFile.readline()
    inputFile.close()
    return fuzzySetsDict

def readRulesFile():
    inputFile = open('Files/Rules.txt', 'r')
    rules = RuleList()
    line = inputFile.readline()
    while line != '':
        rule = Rule()
        line = line.rstrip()
        elementsList = line.split(', ')
        rule.ruleName = elementsList[0]
        rule.consequent = elementsList[1]
        lhs = []
        for i in range(2, len(elementsList), 1):
            lhs.append(elementsList[i])
        rule.antecedent = lhs
        rules.append(rule)
        line = inputFile.readline()
    inputFile.close()
    return rules

def readApplicationsFile():
    inputFile = open('Files/Applications.txt', 'r')
    applicationList = []
    line = inputFile.readline()
    while line != '':
        elementsList = line.split(', ')
        app = Application()
        app.appId = elementsList[0]
        app.data = []
        for i in range(1, len(elementsList), 2):
            app.data.append([elementsList[i], int(elementsList[i+1])])
        applicationList.append(app)
        line = inputFile.readline()
    inputFile.close()
    return applicationList

# ------------HERE STARTS STUDENT'S CODE--------------

# First reads the inputs nd outputs sets, as well as rules and applications from the files given

Inputsets = readFuzzySetsFile('Files/InputVarSets.txt')
Risksets = readFuzzySetsFile('Files/Risks.txt')
rules = readRulesFile()
applications = readApplicationsFile()
final_risk = []

# Split the fuzzy sets in the dictionary and evaluate each of the applications in the corresponding set
for app in applications:
    # STEP1: FUZZIFICATION
    for setId, fuzzy_set in Inputsets.items():
        for var_label, value in app.data:
            if var_label == fuzzy_set.var:
                fuzzy_set.memDegree = skf.interp_membership(fuzzy_set.x, fuzzy_set.y, value)

    # STEP2: RULE EVALUATION
    risk_functions = []
    for rule in rules:
        antecedent_result = []  # here we store membership degree of each antecedent
        for antecedent_setid in rule.antecedent:  # taking the antecedents 1 by 1
            for setId, fuzzy_set in Inputsets.items():
                if antecedent_setid == setId:
                    antecedent_result.append(fuzzy_set.memDegree)
        # Let's compute similarity degree ==  evaluate the conjunction of the rule antecedents (min)
        min_memDegree = 1
        for ant_memDegree in antecedent_result:
            if ant_memDegree < min_memDegree:
                min_memDegree = ant_memDegree
        similarity_degree = min_memDegree
        # Now let's cut the consequent membership function at the level of the antecedent degree
        for setId, riskset in Risksets.items():
            if setId == rule.consequent:
                consequent_result = [rule.consequent, similarity_degree]
                # using clipping
                conseq_membership_function = np.fmin(riskset.y, similarity_degree)
                risk_functions.append(conseq_membership_function)

    # STEP3: COMPOSITION
    # Now we unificate the output of all the rules, using aggregation
    max_result_function = risk_functions[0]
    for function in risk_functions:
        max_result_function = np.fmax(function, max_result_function)

    # Get the output variable range
    x_output = []
    for setId, riskset in Risksets.items():
        x_output = riskset.x

    # STEP4: DEFUZZIFICATION
    # Calculate the membership degree for each risk label using the centroid value
    centroid = fuzz.defuzz(x_output, max_result_function, 'centroid')
    # Save value of centroid (final risk) in a list
    final_risk.append(centroid)

# Save in "Results.txt" the final risk for all applications
with open("Files/Results.txt", "w") as file:
    for app, risk_val in zip(applications, final_risk):
        file.write(f"[Application {app.appId}] Final Risk: {risk_val} \n")

