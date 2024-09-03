#!/bin/bash
echo "Sourcing .bashrc"
source ~/.bashrc
echo "NVM version:"
nvm --version
echo "Available Node.js versions:"
nvm ls
echo "Current Node.js version:"
nvm current