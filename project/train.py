import hydra
from omegaconf import DictConfig

@hydra.main(
  version_base=None,
  config_path="config1",
  config_name="config2"
)
def main(cfg: DictConfig):
  print(cfg)
  print(cfg.training.lr)
  print(cfg.activation)
  print(cfg.toto.name)
  print(cfg.toto.param.channels)
  print(cfg.name)
  print(cfg.param.lr)

if __name__ == "__main__":
  main()
