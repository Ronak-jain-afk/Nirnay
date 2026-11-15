from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

DecisionKind = Literal["noul", "choice", "score"]

class DecisionInput(BaseModel):
    state: str = Field(..., description="Contextual background, environment state, logs, or document")
    question: str = Field(..., description="The query or decision question to evaluate")
    options: List[str] = Field(..., description="Candidate options to choose between")
    kind: DecisionKind = Field(default="choice", description="Decision primitive type: noul, choice, or score")

class DecisionRecord(BaseModel):
    """Schema corresponding to rows in SargeDev/jev-distill-corpus-v3."""
    state: str
    question: str
    options: List[str]
    kind: DecisionKind
    distribution: Dict[str, float] = Field(..., description="Teacher probability distribution across options")
    winner: Optional[str] = None

class DecisionResult(BaseModel):
    decision: str = Field(..., description="Top-1 selected decision label")
    probabilities: Dict[str, float] = Field(..., description="Calibrated probabilities summing to 1.0")
    logits: Dict[str, float] = Field(..., description="Raw model logits before softmax")
    temperature: float = Field(default=1.0, description="Temperature parameter applied during inference")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    kind: DecisionKind
