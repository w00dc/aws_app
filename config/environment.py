"""Establishes shared environment constants for use in the application"""

from functools import lru_cache
from pydantic_settings import BaseSettings

# This file is for defining your environment variables that will be imported from Kubernetes
# Secrets. The variable names are case-insensitive, so you can follow the standard of all caps
# and still import them as lower-case. Any variables required for testing should be defined
# with TEST_ in front.

# Calling the get_settings function from other modules will import your settings for use


class Settings(BaseSettings):
    """Primary settings object for initializing application"""

    port: int = 8501
    app_name: str = "dm_bot"
    log_level: str = "DEBUG"
    openai_api_key: str = "missing"
    vectorstore_path: str = ""

    class Config:
        """Defines how pydantic should read in settings"""

        env_file = ".env"


class TestSettings(Settings):
    """Settings object for test environment. Inherits settings from main Settings object"""

    __test__ = False
    log_level: str = "DEBUG"
    bot_env: str = "dev"

    class Config:
        """Defines how pydantic should read in test settings"""

        env_prefix = "TEST_"
        env_file = ".env"


# Configures environment settings to only be read once
@lru_cache()
def get_settings():
    """Retrieves settings just once"""
    return Settings()  # type: ignore
