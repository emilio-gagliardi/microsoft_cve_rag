from pydantic import BaseModel, Field, model_validator
from typing import Optional
import os


class VectorDBCredentialsSchema(BaseModel):
    username: Optional[str] = Field(None, env="VECTOR_DATABASE_USERNAME")
    password: Optional[str] = Field(None, env="VECTOR_DATABASE_PASSWORD")
    host: str = Field("localhost", env="VECTOR_DATABASE_HOST")
    port: int = Field(6333, env="VECTOR_DATABASE_PORT")
    uri: Optional[str] = Field(None, env="VECTOR_DATABASE_URI")
    protocol: str = Field(None, env="VECTOR_DATABASE_PROTOCOL")

    @model_validator(mode="after")
    def construct_uri(cls, values):
        if not values.uri:
            values.uri = f"{values.protocol}://{values.host}:{values.port}"
        return values

    @model_validator(mode="after")
    def validate_credentials(cls, values):
        environment = os.getenv("ENVIRONMENT", "local")
        if environment.lower() == "production":
            if not values.username or not values.password:
                raise ValueError(
                    "Username and password are required in production environment"
                )
        return values


class GraphDBCredentialsSchema(BaseModel):
    username: str = Field(..., env="GRAPH_DATABASE_USERNAME")
    password: str = Field(..., env="GRAPH_DATABASE_PASSWORD")
    host: str = Field("localhost", env="GRAPH_DATABASE_HOST")
    port: int = Field(7474, env="GRAPH_DATABASE_PORT")
    uri: Optional[str] = Field(None, env="GRAPH_DATABASE_URI")
    protocol: str = Field(None, env="GRAPH_DATABASE_PROTOCOL")

    @model_validator(mode="after")
    def construct_uri(cls, values):
        if not values.uri:
            values.uri = f"{values.protocol}://{values.username}:{values.password}@{values.host}:{values.port}"
        return values


class DocumentsDBCredentialsSchema(BaseModel):
    username: str = Field(..., env="DOCUMENTS_DATABASE_USERNAME")
    password: str = Field(..., env="DOCUMENTS_DATABASE_PASSWORD")
    host: Optional[str] = Field(None, env="DOCUMENTS_DATABASE_HOST")
    port: Optional[int] = Field(None, env="DOCUMENTS_DATABASE_PORT")
    db_cluster: str = Field(..., env="DOCUMENTS_DATABASE_CLUSTER")
    db_cluster_id: str = Field(..., env="DOCUMENTS_DATABASE_CLUSTER_ID")
    protocol: str = Field(..., env="DOCUMENTS_DATABASE_PROTOCOL")
    uri: Optional[str] = Field(None, env="DOCUMENTS_DATABASE_URI")

    @model_validator(mode="after")
    def construct_uri(cls, values):
        if not values.uri:
            values.uri = f"{values.protocol}://{values.username}:{values.password}@{values.db_cluster}.{values.db_cluster_id}.mongodb.net/"
        return values


class SQLDBCredentialsSchema(BaseModel):
    username: str = Field(..., env="SQL_DATABASE_USERNAME")
    password: str = Field(..., env="SQL_DATABASE_PASSWORD")
    host: str = Field("localhost", env="SQL_DATABASE_HOST")
    port: int = Field(27017, env="SQL_DATABASE_PORT")
    uri: Optional[str] = Field(None, env="SQL_DATABASE_URI")
    protocol: str = Field(None, env="SQL_DATABASE_PROTOCOL")

    @model_validator(mode="after")
    def construct_uri(cls, values):
        if not values.uri:
            values.uri = f"{values.protocol}://{values.username}:{values.password}@{values.host}:{values.port}/"
        return values
