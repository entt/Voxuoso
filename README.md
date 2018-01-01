# Voxuoso
A laryngitis mobile detection app, which will use Deep Recurrent Neural Networks (DRNN) and objective acoustic parameters.

## About
This is a ReactNative mobile application served over Flask microframework.

## Requirements
**Backend requirements**
* Python: 2.7
* See [virtual_requirements.txt](https://github.com/entt/Voxuoso/blob/master/backend/virtual_requirements.txt)

**Frontend**
* NodeJS: 6.11.4
* NPM: 3.10.10
* Exp: 46.0.3
* Expo SDK: 24.00
* React: 16.1.1
* ReactNative: 0.50.3

## Build Guide
```
# For frontend
cd frontend
npm install

# For backend
cd ../backend
virtualenv -p python venv
source venv/bin/activate
pip install -r requirements
```

## Running the App
```
# For frontend
cd frontend

# Hot reload, use Expo client to scan QR code
exp start

# If you have a connected Android device use,
# Note: Hot-reload
exp android

# For backend
cd ../backend
export FLASK_APP=app.py
flask app

# Alternatively,
python app.py
```
