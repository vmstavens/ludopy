# config
# NUMBER_MAX_GENERATIONS = 10
# NUMBER_OF_SOLUTIONS    = 12
# NUMBER_OF_RUNS         = 3
NUMBER_MAX_GENERATIONS = 150
NUMBER_OF_SOLUTIONS    = 100
NUMBER_OF_RUNS         = 50


MUTATION_RATE          = 0.05
CROSSOVER_RATE         = 0.5
HOME_POSITION          = 99
CROSSOVER_METHOD       = ["static_cut","probabilistic_cut"][1]
SELECTION_METHOD       = ["tournament","roulette"][0]
TOKENS_ON_GOAL_SCALING = 1