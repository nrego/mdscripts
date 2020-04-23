# Nbr 05/30/19
# (use %paste in IPython)

from __future__ import division, print_function

import numpy as np
import MDAnalysis
from scipy.spatial import cKDTree
from scratch.quiz.lib import *


n_rounds = 4
# shape : n_rounds, n_qs
questions = {
    0: ["How are you doing today?", 
        "What's your favorite color?",
        "Tell me about someone at work that annoys you.",
        "How do you feel about wealth inequality?",
        "Charter schools: Good things or bad things?"
        ],
    # Gray round
    1: ["What is the main character of '50 shades of gray' named?",
        "Name as many animals as you can that have the word 'gray' in their name (up to 3)",
        "Gray-scale pixels are encoded by one number - color by 3. what are the 3 numbers?",
        "Dogs can only see in black and white: true or false?",
        "Name The type of cloud associated with storms."
        ],

    # "General" knowledge
    2: ["In the punic wars, which general suprised romans...",
        "Name as many US companies as you can that have the word 'general' in them (up to 3)",
        "Complete the sentence: 'Hello there!'  'General ____!'",
        "Latin root of 'general' means 'origin, type, group', or 'race'. what is the root?",
        "Famous French generals (or war leaders)",
        "Beyonce debut"
        ],

    # "FOOOD"
    3: ["Shakshuka",
        "Mex natl dish",
        "Durian",
        "masala refers to what?",
        "This famous italian dish is prepared with capers, anchovies, and olives",
        "French mother sauces - what are they??",
        "'Vindaloo' comes from the Portueguese - 'vin' and 'aloo' - what does it mean? one pt each"
        ],

    # Psychological torture
    4: ["Produce a red sock.",
        "Two ropes each takes 1 hr to burn, but not at same rate. how to measure 45 min?",
        "Make me a fucking sandwich",
        "Riddle: I am a chest with no edges, hinges, no key, no lid, yet within me golden treasure's hid. what am i?",
        "You have an aquarium with 200 fish. 99pc of them are red. How many do you have to remove to have 98pc red fish?",
        "Elevator question: man on 10th floor takes it down, only takes it up to 7th floor. why?",
        "I can only be kept if I am given. To some people, I am strong as iron. To others, I am weak as smoke.",
        "Riddle: This thing all things devours: birds, beasts, trees, flowers; gnaws iron, bites steal, grinds hard stone to meal. slays kings, ruins town, and beats high mountain down"]

}



#q1.add_score(1,1)

qm = QuizzMaster()


qm.init_quizzers('responses.csv')
#for q in qm.quizzers.values():
#    q.scores[0] = 5

qm.collect_and_assign(questions, 0, 'responses.csv')
