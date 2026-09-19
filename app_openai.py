import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# 1. Define the structured output format using Pydantic
class ResumeAnalysis(BaseModel):
    match_percentage: int = Field(
        description="ATS match score from 0 to 100 based on the job description"
    )
    matched_skills: list[str] = Field(
        description="Keywords and skills found in the resume that match the job description"
    )
    missing_skills: list[str] = Field(
        description="Important keywords and skills from the job description missing in the resume"
    )
    recommendations: list[str] = Field(
        description="3 actionable bullet points to improve the resume for this job role"
    )


# Streamlit page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer by Venkys.AI")
st.write("Upload your resume and a job description to check your ATS compatibility.")


# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password"
    )
    st.markdown(
        "[Create/manage your OpenAI API key](https://platform.openai.com/api-keys)"
    )


# Main Interface: Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF format only)",
        type=["pdf"]
    )

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area(
        "Paste the Target Job Description Here",
        height=200
    )


# Process analysis when button is clicked
if st.button("Analyze Resume", type="primary"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Extracting text and analyzing with OpenAI..."):
            try:
                # Extract text from PDF
                reader = PdfReader(uploaded_file)
                resume_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

                if not resume_text.strip():
                    st.error(
                        "Could not extract text from this PDF. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                # Initialize OpenAI model through LangChain
                # GPT-5.6 Luna is selected here as a cost-sensitive model.
                llm = ChatOpenAI(
                    model="gpt-5.6-luna",
                    api_key=openai_api_key,
                    temperature=0.2
                )

                # Structured JSON parser
                parser = JsonOutputParser(
                    pydantic_object=ResumeAnalysis
                )

                # Prompt
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert ATS (Applicant Tracking System) "
                        "optimizer. Compare the candidate's resume text against "
                        "the provided job description. Provide an unbiased "
                        "assessment. Return output strictly in JSON format "
                        "matching the schema instructions.\n{format_instructions}"
                    ),
                    (
                        "human",
                        "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"
                    )
                ])

                # Chain: Prompt -> OpenAI -> JSON Parser
                chain = prompt | llm | parser

                # Execute
                result = chain.invoke({
                    "resume": resume_text,
                    "job_description": job_description,
                    "format_instructions": parser.get_format_instructions()
                })

                # Render results
                st.success("Analysis Complete!")
                st.divider()

                score = result.get("match_percentage", 0)

                if score >= 75:
                    st.balloons()
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Strong Match"
                    )
                elif score >= 50:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Needs Improvement",
                        delta_color="off"
                    )
                else:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Weak Match",
                        delta_color="inverse"
                    )

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("✅ Matched Skills & Keywords")
                    for skill in result.get("matched_skills", []):
                        st.markdown(f"- {skill}")

                with res_col2:
                    st.subheader("❌ Missing Critical Keywords")
                    for skill in result.get("missing_skills", []):
                        st.markdown(
                            f"- <span style='color:#ff4b4b'>"
                            f"**{skill}**</span>",
                            unsafe_allow_html=True
                        )

                st.subheader("💡 Recommendations to Optimize Your Resume")
                for rec in result.get("recommendations", []):
                    st.markdown(f"* {rec}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info(
                    "If the error mentions billing, quota, or insufficient "
                    "credits, check your OpenAI API billing/usage settings."
                )
