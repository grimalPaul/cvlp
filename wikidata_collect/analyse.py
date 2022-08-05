import json

with open("data/wikimage.json", "r") as f:
    data = json.load(f)

print(f"nb enitites : {len(data)}")
stats = {}
for id, images in data.items():
    if str(len(images)) in stats:
        stats[str(len(images))] +=1
    else:
        stats[str(len(images))]=1

print(stats)