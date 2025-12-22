import pandas as pd
from player import Player

df = pd.read_csv("data/2026.csv")



allMap = {}
count=0

for name in df['Player']:
    player = Player(name)
    val = player.AllStarProb()
    if val >= .2:
        count+=1
        allMap[name] = val

print(count)
sortedMap = sorted(allMap.items(), reverse=True, key=lambda x:x[1])

count = 1
for name, val in sortedMap:
    print(count,": " ,name, ":", val)
    count+=1
