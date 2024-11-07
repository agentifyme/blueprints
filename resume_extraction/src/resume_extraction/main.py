# Resume Processing Script without Embedding and MongoDB

from agentifyme import workflow, task
import os
import openai
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import openai
import structlog
import docx
import PyPDF2

# Configure structlog for logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models for structured data
class ContactInfo(BaseModel):
    first_name: str
    last_name: str
    middle_name: str
    phone_number: str
    email: str
    linkedin: str

class Education(BaseModel):
    degree: str
    year: str
    college: str

class Skills(BaseModel):
    programming_languages: List[str]
    frameworks_libraries: List[str]
    tools_technologies: List[str]

class Experience(BaseModel):
    company_client: str
    job_role: str
    summary_of_responsibilities: str
    place: str
    start_date: str
    end_date: str

class CandidateData(BaseModel):
    contact: ContactInfo
    education: List[Education]
    skills: Skills
    experience: List[Experience]

# Function to extract text from various resume file formats
def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a resume file.

    Args:
        file_path (str): The path to the resume file.

    Returns:
        str: Extracted text from the resume.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension in ['.doc', '.docx']:
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            logger.info("Extracted text from DOC/DOCX file.")
            return text
        elif file_extension == '.pdf':
            reader = PyPDF2.PdfReader(file_path)
            text = ''
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            logger.info(f"Extracted text from PDF file with {len(reader.pages)} pages.")
            return text
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info("Extracted text from TXT file.")
            return text
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            raise ValueError("Unsupported file format. Please provide a .doc, .docx, .pdf, or .txt file.")
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        raise

# Define the task for processing resume text with OpenAI API
@task(name="Process Resume Text with OpenAI")
def process_resume_text(resume_text: str) -> dict:
    """
    Processes resume text using OpenAI API to extract structured data.

    Args:
        resume_text (str): The plain text content of the resume.

    Returns:
        dict: Parsed resume details as a dictionary.
    """
    logger.info("Sending resume text to OpenAI API for processing.")

    # Define the prompt
    prompt_content = (
        "You are a resume extractor who extracts candidate details. Extract contact information, "
        "education details, skills, and experience from the provided resume text. "
        "Return the data in the following JSON schema: "
        "{'contact': {'first_name': '', 'last_name': '', 'middle_name': 'NA', 'phone_number': 'NA', 'email': 'NA' or None, 'linkedin': 'NA'}, "
        "'education': [{'degree': 'NA', 'year': 'NA', 'college': 'NA'}], "
        "'skills': {'programming_languages': [], 'frameworks_libraries': [], 'tools_technologies': []}, "
        "'experience': [{'company_client': 'NA', 'job_role': 'NA', 'summary_of_responsibilities': 'NA', 'place': 'NA', 'start_date': 'NA','end_date': 'NA'}]}. "
        "Ensure the output is a properly formatted JSON string without any additional delimiters or code block markers."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": resume_text},
            ],
        )

        resume_data_json = response.choices[0].message.content.strip()

        # Parse JSON string to dictionary
        resume_data_dict = json.loads(resume_data_json)
        
        # Use **dict to initialize the Pydantic model
        resume_data = CandidateData(**resume_data_dict)
        logger.info("Resume text processed successfully.")
        
        return resume_data  # Convert Pydantic model to dictionary for output

    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decode error: {json_err}")
        raise
    except Exception as e:
        logger.error(f"Error processing resume text with OpenAI: {e}")
        raise


# Define the workflow
@workflow(name="Extract Resume Workflow")
def extract_resume_workflow(file_path: str) -> dict:
    """
    Workflow to extract structured data from a resume file.

    Args:
        file_path (str): The path to the resume file.

    Returns:
        dict: Parsed resume details.
    """
    logger.info("Starting resume extraction workflow.")

    # Step 1: Extract text from the resume file
    resume_text = extract_text_from_file(file_path)

    # Step 2: Process the extracted text with OpenAI API
    parsed_resume_data = process_resume_text(resume_text)

    logger.info("Resume extraction workflow completed successfully.")

    # Convert Pydantic model to dictionary for output
    return parsed_resume_data.dict()

# Usage:
if __name__ == "__main__":
    # Specify the file path directly
    file_path = "Abdul_S_resume.pdf"  # Replace with the actual path to the resume file

    try:
        # Execute the workflow
        result = extract_resume_workflow(file_path)
        print(json.dumps(result, indent=4))  # Print in formatted JSON
    except Exception as e:
        logger.error(f"Failed to extract resume data: {e}")
        print(f"Error: {e}")


