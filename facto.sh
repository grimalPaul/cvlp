#!/bin/bash

rsync -az -v --exclude 'facto.sh' --exclude 'get_results.sh' --exclude 'results/' --exclude 'sync.sh' --exclude 'test.ipynb' --exclude '__pycache__/' --exclude 'data_model/' --exclude 'IDEA.md'--exclude 'test.ipynb' --exclude 'README.md' --exclude 'factory_sync.sh' --exclude 'note.md' --exclude 'old_note.md' --exclude 'TODO.md' . factoryIA:/home/pgrimal/CVLP

echo "Done!"
