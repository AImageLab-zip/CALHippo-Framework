import tensorflow as tf
from loguru import logger


def limit_tensorflow_vram():
    MAX_TF_VRAM_MB = 5120  # 5GB
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=MAX_TF_VRAM_MB)],
            )
            logger.info("TensorFlow memory growth enabled.")
        except RuntimeError as e:
            logger.error(f"VRAM limit setting failed: {e}")
