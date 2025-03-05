class Config:
    def __init__(self, config_dict: dict):
        """
        Dynamically set attributes based on the provided configuration dictionary.

        Args:
            config_dict (dict): Dictionary loaded from a YAML configuration file.
        """
        self._load_config(config_dict)

    def _load_config(self, config_dict: dict) -> None:
        """
        Recursively load configuration keys and values into the class attributes.

        Args:
            config_dict (dict): Dictionary containing configuration data.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # If the value is a dictionary, create a nested Config object
                setattr(self, key, Config(value))
            else:
                # Otherwise, set the attribute directly
                setattr(self, key, value)

    def get_subnet_insize(self) -> int:
        """
        Get the input size for the subnet based on the transformer checkpoint.

        Returns:
            int: Input size for the subnet.
        """
        if hasattr(self, 'transformer_checkpoint'):
            if self.transformer_checkpoint == self.model_checkpoints.HUBERT:
                return 768
            elif self.transformer_checkpoint == self.model_checkpoints.WAV2VEC2:
                return 768
        raise ValueError("Transformer checkpoint not recognized or not set.")