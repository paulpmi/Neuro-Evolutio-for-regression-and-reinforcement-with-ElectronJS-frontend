import sys
sys.path.insert(0, './Algorithm')
sys.path.insert(0, './data')

from Disection import Algorithm
import listToExcel

data = []
for i in range(len(sys.argv)):
    if i != 0:
        data.append(sys.argv[i])

a = Algorithm(int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), int(data[5]))

if data[6] == 'false':
    data[6] = False
else:
    data[6] = True

#print(data)

a.testRun(test=data[6])
a.population.population[0].testAlgoritm(test=data[6])

input = listToExcel.listToExcel("./UItest.txt")
output = listToExcel.listToExcel("./UItestRealValues.txt")
listToExcel.writeToExcel(output, input, "UIreturnCSV.csv")