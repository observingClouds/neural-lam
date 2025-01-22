# Standard library
import sys

# Third-party
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from loguru import logger


class CustomMLFlowLogger(pl.loggers.MLFlowLogger):
    """
    Custom MLFlow logger that adds functionality not present in the default
    """

    def __init__(self, experiment_name, tracking_uri, run_name):
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        mlflow.start_run(run_id=self.run_id, log_system_metrics=True)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("run_id", self.run_id)

    @property
    def save_dir(self):
        """
        Returns the directory where the MLFlow artifacts are saved
        """
        return "mlruns"

    def log_image(self, key, images, step=None):
        """
        Log a matplotlib figure as an image to MLFlow

        key: str
            Key to log the image under
        images: list
            List of matplotlib figures to log
        step: Union[int, None]
            Step to log the image under. If None, logs under the key directly
        """
        # Third-party
        import botocore
        from PIL import Image

        if step is not None:
            key = f"{key}_{step}"

        # Need to save the image to a temporary file, then log that file
        # mlflow.log_image, should do this automatically, but is buggy
        temporary_image = f"{key}.png"
        images[0].savefig(temporary_image)

        img = Image.open(temporary_image)
        try:
            mlflow.log_image(img, f"{key}.png")
        except botocore.exceptions.NoCredentialsError:
            logger.error("Error logging image\nSet AWS credentials")
            sys.exit(1)
