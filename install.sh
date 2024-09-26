#!/bin/bash

echo "Setting up virtualenv 'venv'. Please use the command 'source venv/bin/activate' to activate the virtualenv."
python -m venv venv
source venv/bin/activate

echo "Installing DiffAE repository..."
git clone https://github.com/phizaz/diffae.git

echo "Switching into diffae project directory."
cd diffae

echo "Installing packages for the DiffAE repo..."
pip install -r requirements.txt
cd ..

echo "Installing additional packages for Greedy-DiM..."
pip install -r requirements.txt

# DiffAE codebase needs to be toplevel due to import hell and abuse of import *
mv diffae/* .
rm -rf diffae

read -p "Enter password for morphing code: " password
gpg --batch --passphrase "$password" -d run_dim.py.gpg > run_dim.py

deactivate
