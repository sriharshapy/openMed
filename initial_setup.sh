python -m venv venv
./venv/Scripts/activate
python -m pip install --upgrade pip
pip install --upgrade setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
deactivate