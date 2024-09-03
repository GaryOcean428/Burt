#!/bin/bash
echo "Current directory:"
pwd
echo "Home directory:"
echo $HOME
echo "NVM directory:"
echo $NVM_DIR
echo "Manually sourcing NVM:"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
echo "NVM version:"
nvm --version
echo "Node.js version:"
node --version
echo "npm version:"
npm --version
echo "PATH:"
echo $PATH