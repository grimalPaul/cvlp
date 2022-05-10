#!/bin/bash

rsync -az --exclude 'data1' --exclude 'sync.sh' --exclude 'test.ipynb' --exclude 'README.md' --exclude 'factory_sync.sh' --exclude 'EXPERIMENTS.md' --exclude 'LICENCE' --exclude 'ANNOTATION.md' . bergamote:/home/pgrimal/ViQuAE

echo "Done!"
