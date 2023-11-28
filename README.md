# Pokemon_RL_Model

This project is a Reinforcement Learning model designed for Pokémon battles. It uses the poke-env library to interact with the Pokémon environment and implements a Deep Q-Network (DQN) for decision-making. The model is trained to understand Pokémon type effectiveness and can be used to simulate battles and strategies in a Pokémon game environment.  
**Requirements**
1. Have an updated version of Node.js installed
2. Have Python 3.11 installed, or a similar version
3. pip install asyncio
4. pip install tabulate
5. pip install numpy
6. pip install keras==2.12.0
7. pip install poke-env==0.7.0
8. pip install tensorflow==2.12.0
9. pip install gym==0.26.2

**Usage**
1. Clone the pokemon showdown source code repo: https://github.com/smogon/pokemon-showdown-client/blob/master/README.md
2. Run the command
```
npm install
```
This should install all dependencies  
3. Run the command 
```
node pokemon-showdown start --no-security
```
This should create a private showdown server on localhost  
4. Clone this repo  
5. Run reinforcement_bot.py  

The file model.h5 is an already trained model. To start training of a new model delete the file. To evaluate the current model just run the reinforcement_bot.py file
