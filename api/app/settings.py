from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    elastic_host: str
    elastic_port: int
    elastic_username: str
    elastic_password: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


settings = Settings()
