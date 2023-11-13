import sys


indata = sys.stdin.read()
print(indata)
gabor = []
rmse = []
for line in indata.strip().split("\n"):
    mode = line[:4]
    if mode == "gabo":
        gabor.append(line)
    elif mode == "rmse":
        rmse.append(line)
    else:
        print(f"line: '{line}'")
        exit(f"Error. I don't recognize the mode. I got '{mode}'")

from matplotlib.pyplot import *

data = {"gabo":{"train":{}, "test":{}},"rmse":{"train":{}, "test":{}} }
for Y in [gabor,rmse]:
    for line in Y:
        lst = "[" + re.search("\[(.*?)\]", line).group().replace(" ", ",").replace(";,","],\n[") + "]"
        words = line.split(" ")
        stat = words[0]
        mode = words[5]
        count = int(words[10]) * int(words[9])
        data[stat][mode][count] = eval(lst)

for stat in data.keys():
    for label in list(data[stat].keys()):
        x = list(data[stat][label].keys())
        idx = np.argsort(x)
        ys = list(data[stat][label].values())
        for i, symmetry in enumerate(["SS", "SP", "PS", "PP"]):
            y = []
            for matrix in ys:
                I,J = (i//2, i%2)
                y.append(matrix[I][J])
            scatter( np.array(x)[idx], np.array(y)[idx], label=symmetry)
            xlabel("Sample size [*]")
            ylab = "Rel. error" if stat == "gabo" else "RMSE"
            ylabel(ylab)
            title(f"{ ylab } for { label.lower() } set")
            yscale("log")
            xscale("log")
            xmin, xmax = np.log10(np.min(x)), np.log10(np.max(x))
            xlim( 10**np.floor(xmin), 10**np.ceil(xmax) )
            legend()
        show()
