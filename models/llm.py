import os
from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL


def get_chatgroq_model():
    """
    Initialize and return the Groq chat model.
    Reads API key from config (which pulls from environment variable).
    """
    try:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please export it: export GROQ_API_KEY='your_key'"
            )

        groq_model = ChatGroq(
            api_key=api_key,
            model=GROQ_MODEL,
            temperature=0.2,
        )
        return groq_model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")