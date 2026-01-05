import os
import dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. API Configuration
# It is recommended to use an environment variable: export GOOGLE_API_KEY="Your_Key"
dotenv.load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


# 2. JSON Structure Definition (Pydantic)
# This class defines the schema that the Gemini model must strictly follow.
class ToxicityAnalysis(BaseModel):
    # Numeric fields corresponding to the requested dataset columns (scale 0.0 - 1.0)
    toxicity: float = Field(
        ...,
        description="General toxicity score (0-1). Is the comment rude, disrespectful, or unreasonable? Does it make people want to leave the discussion?",
    )
    severe_toxicity: float = Field(
        ...,
        description="Severe toxicity score (0-1). Very hateful, aggressive, or violent content.",
    )
    obscene: float = Field(
        ...,
        description="Obscenity score (0-1). Profanity, vulgarity, or offensive language.",
    )
    threat: float = Field(
        ..., description="Threat score (0-1). Suggestions of physical harm or violence."
    )
    insult: float = Field(
        ...,
        description="Insult score (0-1). Disrespectful or inflammatory language towards others.",
    )
    identity_attack: float = Field(
        ...,
        description="Identity attack score (0-1). Attacks based on race, religion, sexual orientation, gender, disability, etc.",
    )
    sexual_explicit: float = Field(
        ...,
        description="Sexually explicit score (0-1). References to sexual acts, body parts, or sexual content.",
    )

    # Additional fields required for qualitative analysis
    deciding_fragments: List[str] = Field(
        ...,
        description="List of specific text fragments (quotes) that were decisive in classifying the text as toxic.",
    )
    ambiguous_fragments: List[str] = Field(
        ...,
        description="Fragments that were difficult to classify, e.g., irony, sarcasm, or context-dependent slang.",
    )
    justification: str = Field(
        ..., description="Detailed reasoning for the classification decisions."
    )


def analyze_text_toxicity(text_fragment: str) -> Optional[ToxicityAnalysis]:
    """
    Sends text to Gemini 2.0 Flash and returns a structured JSON object
    classifying the toxicity levels.

    Args:
        text_fragment (str): The text content to analyze.

    Returns:
        ToxicityAnalysis: A Pydantic object containing scores and reasoning,
                          or None if an API error occurs.
    """
    client = genai.Client(api_key=API_KEY)

    # System prompt instructing the model on its role and output format
    prompt = f"""
    Analyze the following text fragment for toxicity levels.
    You are a precise content moderation classifier.
    
    Text to analyze:
    "{text_fragment}"
    
    Return the result in a JSON format strictly adhering to the defined schema.
    Scores (float) must be between 0.0 and 1.0.
    Provide the justification in English.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",  # Using the latest Flash model available
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ToxicityAnalysis,  # Passing the Pydantic schema
                temperature=0.0,  # 0.0 for maximum determinism and consistency
            ),
        )

        # The SDK automatically validates and parses the JSON into our Pydantic object
        return response.parsed

    except Exception as e:
        print(f"Error communicating with the API: {e}")
        return None
