import csv
import math


def listToExcel(filename):

    f = open(filename).readlines()

    header = ["Output" + str(i) for i in range(len(f))]
    header = ["Real"] + header

    #print(f)
    for i in f:

        #print(len(i.split(',')))

        data = []
        #print(len(i.split(',')))
        for j in i.split(','):
            number = ""
            for k in j:
                if k == '-':
                    number += k
                if k != '.':
                    try:
                        float(k)
                        number += k
                    except ValueError:
                        pass
                else:
                    number += '.'
            try:
                data.append(float(number))
            except ValueError:
                pass

        #print(data)
        alldata = []

        alldata.append(data)
        finalValues = zip(*alldata)
        with open('outputTest.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            writer = csv.DictWriter(myfile, fieldnames=header)
            writer.writeheader()
            wr = csv.writer(myfile)
            wr.writerows(finalValues)

        #virtualOutput = [math.sin(i) for i in range(100)]
        #alldata.append(virtualOutput)
        #finalValues = zip(*alldata)

        return data


def writeToExcel(input, output, outputFilename):
    alldata = []

    alldata.append(input)
    alldata.append(output)
    finalValues = zip(*alldata)

    header = ["Output"]
    header = ["Start"] + header

    with open(outputFilename, 'w', encoding="ISO-8859-1", newline='') as myfile:
        writer = csv.DictWriter(myfile, fieldnames=header)
        writer.writeheader()
        wr = csv.writer(myfile)
        wr.writerows(finalValues)


#input = listToExcel("./test.txt")
#output = listToExcel("./testRealValues.txt")
#writeToExcel(output, input, "returnCSV.csv")
