from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    elastic_host: str
    elastic_port: int
    elastic_username: str
    elastic_password: str
    elastic_veryify_cert: bool = False

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


settings = Settings()
