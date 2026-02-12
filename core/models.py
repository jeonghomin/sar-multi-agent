"""Pydantic 모델 정의"""
from pydantic import BaseModel, Field
from typing import Literal


class RouterQuery(BaseModel):
    datasource: Literal["casual_chat", "retrieval", "vision", "sar_processing"] = Field(
        ...,
        description="Given a user question choose to route it to casual_chat, retrieval, vision, or sar_processing."
    )


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search", "extract_coordinates"] = Field(
        ...,
        description="Given a user question choose to route it to web search, extract_coordinates, or a vectorstore."
    )


class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer 'yes' if the document is relevant to the question, otherwise answer 'no'."
    )


class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer 'yes' if the generation is hallucinated, otherwise answer 'no'."
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question."
    )


class VisionAgent(BaseModel):
    task: Literal["segmentation", "classification", "detection"] = Field(
        ...,
        description="Given a user question, choose which vision task to perform."
    )
