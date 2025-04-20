from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    HUGGINGFACE_API_TOKEN: str
    MODEL_REPO_ID: str
    MODEL_MAX_NEW_TOKENS: int
    MODEL_TEMPERATURE: float
    MODEL_EMBEDDINGS_MODEL_NAME: str
    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), extra="ignore")

    def get_api_token(self):
        return self.HUGGINGFACE_API_TOKEN
    
    def get_model_repo_id(self):
        return self.MODEL_REPO_ID

    def get_max_new_tokens(self):
        return self.MODEL_MAX_NEW_TOKENS
    
    def get_temperature(self):
        return self.MODEL_TEMPERATURE
    
    def get_embeddings_model_name(self):
        return self.MODEL_EMBEDDINGS_MODEL_NAME

settings = Settings()
