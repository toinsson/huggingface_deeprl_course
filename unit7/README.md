# SETUP

From the hugginface course page, we get the following instructions.
First, create an environement:

```
conda create --name hf_drl_unit7 python=3.9
conda activate hf_drl_unit7
```

Clone the ml-agents branch and install:

```
git clone --branch aivsai https://github.com/huggingface/ml-agents
cd ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents

pip install protobuf==3.20.*
```

Finally, install pytorch:

```
pip install torch
```

We use a conda environement for this.


## training

Then, you can launch the training:

```
mlagents-learn ./ml-agents/config/poca/SoccerTwos.yaml --env=./SoccerTwos/SoccerTwos.x86_64 --run-id="SoccerTwos" --no-graphics
```

To push a model to hub:

```
mlagents-push-to-hf  --run-id="SoccerTwos_1" --local-dir="./results/SoccerTwos" --repo-id="toinsson/poca-SoccerTwos" --commit-message="First Push" --hugginface-token XXX
```

We want to train against specific agents that are placed high in the leaderboard.
For that we are using the asymmetrical features of the mlagents library.
https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/ML-Agents-Overview.md
