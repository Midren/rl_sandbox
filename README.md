## RL sandbox

## Run

Build docker:
```sh
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER_NAME=$USER -t dreamer .
```

Run docker with tty:
```sh
docker run --gpus 'all' -it --rm -v `pwd`:/home/$USER/rl_sandbox -w /home/$USER/rl_sandbox dreamer zsh
```

Run training inside docker on gpu 0:
```sh
docker run --gpus 'device=0' -it --rm -v `pwd`:/home/$USER/rl_sandbox -w /home/$USER/rl_sandbox dreamer python3 rl_sandbox/train.py
```
