# For SALE: State-Action Representation Learning for Deep Reinforcement Learning

Official implementation of the TD7 algorithm. If you use our code please cite the [paper](https://arxiv.org/abs/2306.02451).

### Usage

Example online RL:
```
python main.py --env HalfCheetah-v4
```
Example offline RL:
```
python main.py --offline --env halfcheetah-medium-v2 
```

### Software

Results were originally collected with:
- [Gym 0.25.0](https://github.com/openai/gym)
- [MuJoCo 2.3.3](https://github.com/deepmind/mujoco)
- [Pytorch 2.0.0](https://pytorch.org)
- [Python 3.9.13](https://www.python.org)

### Bibtex

```bibtex
@article{fujimoto2023sale,
  title={For SALE: State-Action Representation Learning for Deep Reinforcement Learning},
  author={Fujimoto, Scott and Chang, Wei-Di and Smith, Edward J and Gu, Shixiang Shane and Precup, Doina and Meger, David},
  journal={arXiv preprint arXiv:2306.02451},
  year={2023}
}
```
