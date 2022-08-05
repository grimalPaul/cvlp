import os
import json


with open('data/command?.json') as f:
    command = json.load(f)

os.chdir("data")
os.chdir("Commons_wikimage")
print(os.getcwd())
list_command = command['command']
for c in list_command:
    os.system(c)