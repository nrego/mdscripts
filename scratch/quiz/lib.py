# Nbr 05/30/19
# (use %paste in IPython)

from __future__ import division, print_function

import numpy as np
import MDAnalysis
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

## Keeps track of team name, score
class Quizz:

    def __init__(self, team_name):
        self.name = team_name

        self.scores = np.zeros(4)


    def get_scores(self):
        return self.scores

    def get_score(self, round):
        return self.scores[round]

    def get_score_upto(self, round):
        return np.sum(self.scores[round])

    def add_score(self, round, inc):
        self.scores[round] += inc


## Holds all the quizzers, can plot all scores on the fly
class QuizzMaster:

    def __init__(self):
        self.quizzers = dict()

    def add_quizzer(self, quizzer):
        self.quizzers[quizzer.name] = quizzer

    # Load in round 0, init quizzers
    def init_quizzers(self, infile, sep=','):
        #txt = np.loadtxt(infile, dtype=str)
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        for in_str in lines[1:]:
            #in_str = in_str[0]
            splits = in_str.split(sep)
            team_name = splits[1]
            print("found team name : {}. adding...".format(team_name))

            q = Quizz(team_name)
            self.add_quizzer(q)

    def collect_and_assign(self, questions, round, infile, sep=','):

        this_qs = questions[round]

        with open(infile, 'r') as fin:
            lines = fin.readlines()

        for in_str in lines[1:]:
            #in_str = in_str[0]
            splits = in_str.split(sep)
            team_name = splits[1]
            self.print_names()
            team_idx = int(input("select team for {} (-1 to skip):  ".format(team_name)))
            if team_idx == -1:
                continue

            team_name = list(self.quizzers.keys())[team_idx]
            this_q = self.quizzers[team_name]
            print(" selected team : {}".format(team_name))
            

            ## Now loop thru q's
            for iq, question in enumerate(questions[round]):
                answer = splits[iq+2]
                print("Q: {}  [A: {}]".format(question, answer))

                points = int(input("    points? : "))
                this_q.add_score(round, points)

    def print_names(self):

        for i, name in enumerate(self.quizzers.keys()):
            print('{:02d}: {}\n'.format(i, name))


    # Sort quizzers by score, plot 
    def sort_and_plot(self, round):

        this_quizzers = np.array(list(self.quizzers.keys()))
        this_scores = np.array([q.get_score_upto(round) for q in self.quizzers.values()])
        print("Scores: {}".format(this_scores))
        sort_idx = np.argsort(this_scores)[::-1]
        indices = np.arange(len(this_quizzers))

        fig, ax = plt.subplots()
        ax.bar(indices, this_scores[sort_idx], width=1)
        ax.set_xticks(indices)
        ax.set_xticklabels(this_quizzers[sort_idx])

        plt.show()

