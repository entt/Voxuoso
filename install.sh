# Install virtual environment
echo "Installing virtual environment..."
cd backend
virtualenv -p python venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
pip install -r tensorflow_requirements.txt

# Install node modules
echo "Installing node modules... This will take a while. Have a coffee."
cd ../frontend
npm install
cd ..
