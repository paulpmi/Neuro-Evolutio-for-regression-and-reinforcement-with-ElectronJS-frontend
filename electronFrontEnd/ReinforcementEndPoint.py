import sys
sys.path.insert(0, './Algorithm')
sys.path.insert(0, './data')

from ReinforcementAgent import iteration

data = []
for i in range(len(sys.argv)):
    if i != 0:
        data.append(sys.argv[i])

iteration(int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), int(data[5]), data[6])