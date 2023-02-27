mlagents-learn cfg_base.yaml --env=../SoccerTwos/SoccerTwos.x86_64 --run-id="Base" --no-graphics

mlagents-push-to-hf  --run-id="Base" --local-dir="./results/Base" --repo-id="toinsson/poca-SoccerTwos-Base" --commit-message="First Push"