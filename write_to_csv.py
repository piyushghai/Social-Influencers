import os


def writeToCSV(filename, probs):
    line = 'Id,Choice'
    f1 = open(filename, 'w+')
    f1.write(line)
    f1.write(os.linesep)
    i = 1
    for p in probs:
        f1.write(str(i) + ',' + str(p) + os.linesep)
        i += 1
    f1.close()
