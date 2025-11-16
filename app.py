import google.generativeai as genai
import json
import PyPDF2
import docx
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from google.generativeai.types import GenerationConfig
import numpy as np
import zipfile
import traceback

# --- IMPORT FIX & ADDITION ---
# Added streamlit-agraph for graph visualization
from streamlit_agraph import agraph, Node, Edge, Config


# --- Helper Functions ---

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two 1D numpy arrays."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return ""


def read_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file.name}: {e}")
        return ""


def load_text_from_file(uploaded_file):
    """Loads text from PDF, DOCX, or TXT file."""
    if uploaded_file.name.endswith('.pdf'):
        return read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        return read_docx(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.error(f"Unsupported file format: {uploaded_file.name}")
        return ""


def generate_updated_resume(resume_text, match_analysis):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Custom styles
    header_style = styles['Heading1']
    header_style.fontSize = 16
    header_style.spaceAfter = 18
    header_style.textColor = colors.HexColor('#1a1a1a')

    section_header_style = ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=12,
        textColor=colors.HexColor('#0d47a1'),
        underlineWidth=1,
        underlineOffset=-3
    )

    normal_style = ParagraphStyle(
        name='NormalText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        name='BulletStyle',
        parent=normal_style,
        bulletFontName='Helvetica',
        bulletFontSize=8,
        bulletIndent=10,
        leftIndent=20
    )

    recommendation_style = ParagraphStyle(
        name='RecommendationStyle',
        parent=normal_style,
        fontSize=9,
        textColor=colors.HexColor('#00695c'),
        leftIndent=25,
        spaceAfter=4
    )

    content = []
    content.append(Paragraph("Updated Resume", header_style))
    content.append(Spacer(1, 12))

    # Resume Content Parsing
    resume_parts = resume_text.split("\n")
    current_section = ""
    bullets = []

    def flush_bullets():
        for bullet in bullets:
            content.append(Paragraph(f"• {bullet.strip()}", bullet_style))
        bullets.clear()

    common_sections = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS', 'SUMMARY', 'OBJECTIVE']

    for line in resume_parts:
        line = line.strip()
        if not line:
            continue

        is_section = line.isupper() or any(section in line.upper() for section in common_sections)

        if is_section:
            flush_bullets()
            current_section = line
            content.append(Spacer(1, 12))
            content.append(Paragraph(current_section, section_header_style))
        else:
            bullets.append(line)

    flush_bullets()

    # ATS Recommendations
    if match_analysis.get('ats_optimization_suggestions'):
        content.append(Spacer(1, 20))
        content.append(Paragraph("ATS Optimization Recommendations", section_header_style))
        content.append(Spacer(1, 10))

        for suggestion in match_analysis.get('ats_optimization_suggestions', []):
            section = suggestion.get('section', '')
            current = suggestion.get('current_content', '')
            suggested = suggestion.get('suggested_change', '')
            keywords = ', '.join(suggestion.get('keywords_to_add', []))
            formatting = suggestion.get('formatting_suggestion', '')
            reason = suggestion.get('reason', '')

            content.append(Paragraph(f"• Section: {section}", recommendation_style))
            if current:
                content.append(Paragraph(f"  Current: {current}", recommendation_style))
            content.append(Paragraph(f"  Suggestion: {suggested}", recommendation_style))
            if keywords:
                content.append(Paragraph(f"  Keywords to Add: {keywords}", recommendation_style))
            if formatting:
                content.append(Paragraph(f"  Formatting: {formatting}", recommendation_style))
            if reason:
                content.append(Paragraph(f"  Reason: {reason}", recommendation_style))
            content.append(Spacer(1, 6))

    doc.build(content)
    buffer.seek(0)
    return buffer


# --- NEW HELPER FUNCTIONS FOR GRAPH VIZ ---

def create_agraph_config(height: int = 400, width: int = 500):
    """Creates a reusable Config object for streamlit_agraph with bug fixes and interactions."""
    return Config(
        width=width,
        height=height,
        node={'labelProperty': 'label', 'color': '#90CAF9', 'size': 250},
        edge={'color': '#E0E0E0', 'strokeWidth': 2, 'smooth': {'enabled': True, 'type': 'dynamic'}},
        directed=True,
        # --- FIX: Set collapsible to False to prevent disappearing on click ---
        collapsible=False,
        physics={
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.5
            },
            "solver": "barnesHut",
            "stabilization": {
                "enabled": True,
                "iterations": 1000,
                "updateInterval": 25,
                "fit": True
            }
        },
        # --- Explicitly enable interaction options ---
        interaction={
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "multiselect": False,
            "hover": True,
            "tooltipDelay": 300
        },
        # --- Add navigation buttons for zoom/pan (optional, as mouse is primary) ---
        # Note: 'navigationButtons': True will add default buttons for zoom/pan/reset in the graph canvas
        # but sometimes requires manual CSS adjustments for placement in Streamlit.
        # For now, we rely on mouse interaction and explicit filter options.
        navigationButtons=False
    )


def convert_json_to_agraph(graph_json: dict, node_types: dict, selected_node_types: list = None):
    """
    Converts the API's JSON graph to streamlit_agraph's format,
    with optional filtering by selected_node_types.
    """
    nodes = []
    edges = []

    # Use a set to track added node IDs, as 'id' is the label
    added_nodes = set()

    # Filter nodes first
    filtered_node_labels = set()
    if 'nodes' in graph_json:
        for node in graph_json['nodes']:
            node_id = node.get('label') or node.get('id')
            node_type = node.get('type', 'Unknown').upper()

            # If filters are active, only add nodes of selected types
            if selected_node_types and node_type not in selected_node_types:
                continue

            if node_id and node_id not in added_nodes:
                color = node_types.get(node_type, '#9E9E9E')  # Default color
                nodes.append(Node(id=node_id, label=node_id, color=color, title=node_type))
                added_nodes.add(node_id)
                filtered_node_labels.add(node_id)  # Track labels that made it through filter

    if 'edges' in graph_json:
        for edge in graph_json['edges']:
            source = edge.get('source')
            target = edge.get('target')
            label = edge.get('label')

            # Only add edges where both source and target nodes are in our filtered set
            if source in filtered_node_labels and target in filtered_node_labels and label:
                edges.append(Edge(source=source, target=target, label=label))

    return nodes, edges


# --- JSON Schemas for API Calls ---

# Schema for Graph Generation (used internally for parsing, not as response_schema)
GRAPH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "nodes": {
            "type": "ARRAY",
            "items": {"type": "OBJECT"}
        },
        "edges": {
            "type": "ARRAY",
            "items": {"type": "OBJECT"}
        }
    },
    "required": ["nodes", "edges"]
}

# This schema is used by BOTH simple and graph-based 1-to-1 analysis
MATCH_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall_match_percentage": {"type": "STRING"},
        "matching_skills": {
            "type": "ARRAY",
            "items": {"type": "OBJECT", "properties": {
                "skill_name": {"type": "STRING"},
                "is_match": {"type": "BOOLEAN"}
            }}
        },
        "missing_skills": {
            "type": "ARRAY",
            "items": {"type": "OBJECT", "properties": {
                "skill_name": {"type": "STRING"},
                "is_match": {"type": "BOOLEAN"},
                "suggestion": {"type": "STRING"}
            }}
        },
        "skills_gap_analysis": {
            "type": "OBJECT",
            "properties": {
                "technical_skills": {"type": "STRING"},
                "soft_skills": {"type": "STRING"}
            }
        },
        "experience_match_analysis": {"type": "STRING"},
        "education_match_analysis": {"type": "STRING"},
        "recommendations_for_improvement": {
            "type": "ARRAY",
            "items": {"type": "OBJECT", "properties": {
                "recommendation": {"type": "STRING"},
                "section": {"type": "STRING"},
                "guidance": {"type": "STRING"}
            }}
        },
        "ats_optimization_suggestions": {
            "type": "ARRAY",
            "items": {"type": "OBJECT", "properties": {
                "section": {"type": "STRING"},
                "current_content": {"type": "STRING"},
                "suggested_change": {"type": "STRING"},
                "keywords_to_add": {"type": "ARRAY", "items": {"type": "STRING"}},
                "formatting_suggestion": {"type": "STRING"},
                "reason": {"type": "STRING"}
            }}
        },
        "key_strengths": {"type": "STRING"},
        "areas_of_improvement": {"type": "STRING"}
    },
    "required": [
        "overall_match_percentage", "matching_skills", "missing_skills",
        "skills_gap_analysis", "experience_match_analysis", "education_match_analysis",
        "recommendations_for_improvement", "ats_optimization_suggestions",
        "key_strengths", "areas_of_improvement"
    ]
}


class JobAnalyzer:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            # Model for simple, fast, repetitive tasks
            self.flash_model = genai.GenerativeModel('gemini-2.5-flash')
            # Model for complex reasoning, graph generation, and deep analysis
            self.pro_model = genai.GenerativeModel('gemini-2.5-pro')

            # --- EMBEDDING MODEL FIX ---
            # genai.embed_content expects the model name as a STRING,
            # not a GenerativeModel instance.
            self.embedding_model_name = 'models/text-embedding-004'

        except Exception as e:
            st.error(f"Error configuring Gemini: {e}")
            raise

    # --- 1. Simple (Flat JSON) Methods ---
    # These functions use the FLASH model for fast, simple analysis.

    def analyze_job_simple(self, job_description: str) -> dict:
        """Uses FLASH model for simple flat JSON analysis of a job."""
        prompt = """
        Analyze this job description and provide a detailed JSON with:
        1. Key technical skills required 
        2. Soft skills required
        3. Years of experience required
        4. Education requirements
        5. Key responsibilities
        6. Job level (entry, mid, senior) 

        Respond ONLY with a valid JSON object.
        Job Description:
        {description}
        """
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        try:
            response = self.flash_model.generate_content(
                prompt.format(description=job_description),
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error in simple job analysis: {str(e)}")
            return {}

    def analyze_resume_simple(self, resume_text: str) -> dict:
        """Uses FLASH model for simple flat JSON analysis of a resume."""
        prompt = """
        Analyze this resume and provide a detailed JSON with:
        1. Technical skills
        2. Soft skills
        3. Years of experience
        4. Education details
        5. Key achievements
        6. Core competencies 

        Respond ONLY with a valid JSON object.
        Resume:
        {resume}  
        """
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        try:
            response = self.flash_model.generate_content(
                prompt.format(resume=resume_text),
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error in simple resume analysis: {str(e)}")
            return {}

    def analyze_match_simple(self, job_analysis: dict, resume_analysis: dict) -> dict:
        """
        Uses FLASH model for 1-to-1 simple analysis.
        Returns the standard MATCH_ANALYSIS_SCHEMA.
        """
        prompt = f"""
        You are a professional resume analyzer. Compare the provided job requirements
        and resume details.

        Job Requirements:
        {json.dumps(job_analysis, indent=2)}

        Resume Details:
        {json.dumps(resume_analysis, indent=2)}

        Based on this flat JSON data, perform a match analysis and generate a response
        following this EXACT JSON schema:
        {json.dumps(MATCH_ANALYSIS_SCHEMA, indent=2)}
        """

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=MATCH_ANALYSIS_SCHEMA,
            temperature=0.2
        )
        try:
            response = self.flash_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error analyzing simple match: {e}")
            return {}

    # --- 2. Graph RAG (Nodes/Edges) Methods ---
    # These functions use the PRO model for complex, high-reasoning tasks.

    def _generate_graph(self, text: str, prompt_template: str) -> dict:
        """Internal helper to generate graph using the PRO model."""

        # We rely on the prompt, not a schema, for this complex generation
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        try:
            response = self.pro_model.generate_content(
                prompt_template.format(text=text),
                generation_config=generation_config
            )
            # Check for safety blocks or empty responses
            if not response.text:
                if response.prompt_feedback:
                    st.error(f"Graph generation blocked by safety settings: {response.prompt_feedback}")
                else:
                    st.error("Graph generation failed: Model returned an empty response.")
                return {}

            # Clean up the response text, removing markdown backticks if they exist
            response_text = response.text.strip().lstrip("```json").rstrip("```")
            return json.loads(response_text)

        except json.JSONDecodeError as je:
            st.error(f"Error decoding graph JSON: {je}")
            print(f"Debug - Invalid JSON from model: {response.text}")
            return {}
        except Exception as e:
            st.error(f"Error generating graph: {e}")
            print(f"Debug - Full traceback for _generate_graph:")
            traceback.print_exc()
            return {}

    def analyze_resume_graph(self, resume_text: str) -> dict:
        """Uses PRO model to create a rich resume knowledge graph."""
        # --- FIX: Removed 'Metric' from NODE TYPES and Example ---
        prompt = """
            You are a Knowledge Graph Extractor. Analyze this resume and create a rich graph
            of nodes and edges.

            **NODE TYPES:**
            'Person', 'Company', 'JobTitle', 'Skill', 'Degree', 'University', 'Project', 'Certification', 'Metric'

            **EDGE LABELS:**
            'HELD_ROLE', 'WORKED_AT', 'USED_SKILL', 'EARNED', 'FROM_UNIVERSITY', 'WORKED_ON', 'ACHIEVED', 'HAS_SKILL'

            **IMPORTANT RULE:** The 'id' for a node should be a unique identifier (e.g., 'skill_1', 'job_1'). The 'label' for a node **MUST** be the actual text extracted from the document (e.g., 'Python', 'Software Engineer', 'Increased sales by 20%').

            **Resume Text:**
            {text}

            Respond ONLY with a valid JSON object matching the schema: {{"nodes": [...], "edges": [...]}}
            """
        return self._generate_graph(resume_text, prompt)

    def analyze_job_graph(self, job_text: str) -> dict:
        """Uses PRO model to create a rich job knowledge graph."""
        # FIX APPLIED: Escaped curly braces {{...}} for the schema
        prompt = """
            You are a Knowledge Graph Extractor. Analyze this job description and create a rich graph
            of nodes and edges.

            **NODE TYPES:**
            'Role', 'Company', 'RequiredSkill', 'PreferredSkill', 'RequiredExperience', 'RequiredDegree', 'Responsibility'

            **EDGE LABELS:**
            'REQUIRES_SKILL', 'PREFERS_SKILL', 'REQUIRES_EXPERIENCE', 'REQUIRES_DEGREE', 'HAS_RESPONSIBILITY', 'AT_COMPANY'

            **IMPORTANT RULE:** The 'id' for a node should be a unique identifier (e.g., 'skill_1', 'role_1'). The 'label' for a node **MUST** be the actual text extracted from the document (e.g., 'Python', 'Senior Software Engineer').

            **Job Description Text:**
            {text}

            Respond ONLY with a valid JSON object matching the schema: {{"nodes": [...], "edges": [...]}}
            """
        return self._generate_graph(job_text, prompt)

    def analyze_match_graph(self, resume_graph: dict, job_graph: dict) -> dict:
        """
        Uses PRO model for 1-to-1 Graph RAG analysis.
        Returns the standard MATCH_ANALYSIS_SCHEMA.
        """

        # Check for empty graphs, which cause the "insufficient data" error
        if not resume_graph or not job_graph:
            st.error("Cannot analyze match: Input graph(s) are empty.")
            return {}

        prompt = f"""
        You are a professional resume graph analyzer. Compare the two knowledge graphs provided:
        one for the Job Description (JD) and one for the Resume.

        By traversing these graphs, perform a detailed comparative analysis.

        **Job Description Graph:**
        {json.dumps(job_graph, indent=2)}

        **Resume Graph:**
        {json.dumps(resume_graph, indent=2)}

        **Analysis Steps:**
        1.  **Skills Match:** Find 'RequiredSkill' nodes in the JD. Check if corresponding 'Skill'
            nodes exist in the Resume AND are connected via 'USED_SKILL' to a 'JobTitle'.
        2.  **Experience Match:** Compare 'RequiredExperience' in JD with the sum of 'HAD_DURATION'
            edges from 'JobTitle' nodes in the Resume.
        3.  **Gap Analysis:** Identify key nodes/edges in the JD graph missing from the Resume graph.
        4.  **Strengths:** Identify nodes/edges in the Resume graph that are not required but add value.

        Now, generate a response following this EXACT JSON schema:
        {json.dumps(MATCH_ANALYSIS_SCHEMA, indent=2)}
        """

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=MATCH_ANALYSIS_SCHEMA,
            temperature=0.2
        )

        try:
            response = self.pro_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error analyzing graph match: {e}")
            return {}

    # --- 3. Batch RAG (Embedding & Funnel) Methods ---
    # These functions combine FLASH and PRO models for an efficient 1-to-Many funnel.

    def get_embedding(self, text: str) -> list:
        """Generates embeddings for a given text."""
        try:
            # --- EMBEDDING MODEL FIX ---
            # Pass the model name STRING, not the GenerativeModel instance
            result = genai.embed_content(model=self.embedding_model_name, content=text)
            return result['embedding']
        except Exception as e:
            # The error message from the user's screenshot will be caught here
            st.error(f"Error generating embedding: {e}")
            return []

    def summarize_text_for_embedding(self, text: str) -> str:
        """Uses FLASH model for fast summarization of JDs for embedding."""
        prompt = """
        You are an elite text summarizer. Extract ONLY the core entities from this job description
        that are relevant for matching a resume. 
        Focus on: 
        1. Job Title
        2. Core required skills (max 10)
        3. Years of experience
        4. Core responsibilities (summarized)

        Combine these into a single, dense paragraph.

        TEXT:
        {text}
        """
        try:
            response = self.flash_model.generate_content(prompt.format(text=text))
            return response.text
        except Exception as e:
            st.error(f"Error summarizing JD text: {e}")
            return ""

    def summarize_graph_for_embedding(self, graph: dict) -> str:
        """Uses FLASH model to summarize a resume graph for embedding."""

        if not graph:
            st.error("Cannot summarize: Input graph is empty.")
            return ""

        prompt = """
        You are an elite graph summarizer. Convert this resume knowledge graph (nodes and edges)
        into a single, dense paragraph describing the candidate's profile.
        Focus on:
        1. Job Titles and Companies
        2. Core Skills used in those jobs
        3. Total years of experience
        4. Degrees and Certifications

        GRAPH:
        {graph_json}
        """
        try:
            response = self.flash_model.generate_content(
                prompt.format(graph_json=json.dumps(graph))
            )
            return response.text
        except Exception as e:
            st.error(f"Error summarizing resume graph: {e}")
            return ""

    def analyze_top_matches(self, resume_graph: dict, top_3_jobs: list) -> dict:
        """
        Uses PRO model for 1-to-Many RAG analysis.
        Compares resume graph to top 3 job graphs.
        Returns a NEW JSON structure for the batch UI.
        """

        if not resume_graph:
            st.error("Cannot analyze top matches: Resume graph is empty.")
            return {}

        # Dynamically build the prompt with Top 3 job graphs
        job_graphs_prompt = ""
        for i, job in enumerate(top_3_jobs, 1):
            if job['graph']:  # Only add if graph generation was successful
                job_graphs_prompt += f"\n--- JOB {i} ({job['file_name']}) ---\n"
                job_graphs_prompt += json.dumps(job['graph'], indent=2)
                job_graphs_prompt += "\n"

        if not job_graphs_prompt:
            st.error("Cannot analyze top matches: No valid job graphs were generated for top 3.")
            return {}

        prompt = f"""
        You are a top-tier career analyst. A candidate has provided their resume, and we have
        found the top 3 job descriptions that semantically match their profile from a large batch.

        Your task is to perform a deep, graph-based comparison between the candidate's resume
        graph and EACH of the 3 job graphs.

        **Candidate's Resume Graph:**
        {json.dumps(resume_graph, indent=2)}

        **Top 3 Matching Job Graphs:**
        {job_graphs_prompt}

        **Analysis Request:**
        1.  For each of the 3 jobs, provide a "Match Score" (e.g., "90%") for each.
        2.  Provide a "Match Rationale" explaining *why* it's a good/bad fit, based on
            contextual graph matches (e.g., "Strong match on 'Python' skill, which was
            USED_SKILL at their 'Software Engineer' role, matching the JD's requirement.").
        3.  Provide a "Key Gaps" list for each job.
        4.  Finally, provide a "TopRecommendation" by selecting which of the 3 jobs is
            the absolute best fit and why.

        **Respond ONLY with a valid JSON object following this schema:**
        {{
            "best_match": {{
                "file_name": "job1.pdf",
                "match_score": "95%",
                "rationale": "This is the strongest fit because...",
                "key_gaps": ["Missing 'Kubernetes' experience."]
            }},
            "other_matches": [
                {{
                    "file_name": "job2.pdf",
                    "match_score": "80%",
                    "rationale": "Good skill overlap, but lacks required 5 years of management.",
                    "key_gaps": ["Missing 'Management' experience", "Missing 'AWS' certification"]
                }},
                {{
                    "file_name": "job3.pdf",
                    "match_score": "75%",
                    "rationale": "Matches on junior-level skills, but this is a senior role.",
                    "key_gaps": ["Missing '5+ years' experience", "Missing 'Terraform'"]
                }}
            ]
        }}
        """

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3
        )

        try:
            response = self.pro_model.generate_content(
                prompt, generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error analyzing top matches: {e}")
            return {}


class CoverLetterGenerator:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            # Cover letter generation is a high-reasoning task
            self.pro_model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            st.error(f"Error configuring Gemini: {e}")
            raise

    def generate_cover_letter(self, job_details: str, resume_details: str, match_analysis: dict,
                              tone: str = "professional") -> str:
        """
        Generates a cover letter using the PRO model.
        It works with match_analysis from EITHER the simple or graph-based method.
        """

        if not match_analysis:
            st.error("Cannot generate cover letter: No match analysis provided.")
            return ""

        prompt = """
        Generate a compelling cover letter using this rich analysis.
        The analysis was generated by comparing a resume and a job description.
        Use the 'key_strengths', 'matching_skills', and 'experience_match_analysis'
        to build a strong, evidence-based narrative.

        Job Details:
        {job}

        Candidate Resume Details:
        {resume}

        Match Analysis: 
        {match}

        Tone: {tone}

        Requirements:
        1.  Make it personal and specific, using 'key_strengths'.
        2.  Highlight the strongest matches from 'matching_skills'.
        3.  Address potential gaps (from 'areas_of_improvement') professionally if necessary.
        4.  Keep it concise but impactful.
        5.  Use the specified tone: {tone}
        6.  Add a strong call to action.
        """

        generation_config = GenerationConfig(temperature=0.7)

        try:
            response = self.pro_model.generate_content(
                prompt.format(
                    job=job_details,
                    resume=resume_details,
                    match=json.dumps(match_analysis, indent=2),
                    tone=tone
                ),
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            st.error(f"Error generating cover letter: {e}")
            return ""


# --- Main Application UI ---

def display_1_to_1_ui(match_analysis, resume_text, job_desc_text, cover_letter_gen):
    """Helper function to render the 5-tab UI for 1-to-1 modes."""

    if not match_analysis:
        st.error("Failed to get analysis results. Cannot display UI.")
        return

    st.header("Analysis Results 📊")

    # Match Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Overall Match 🎯",
            f"{match_analysis.get('overall_match_percentage', '0%')}"
        )
    with col2:
        st.metric(
            "Skills Match 🧠",
            f"{len(match_analysis.get('matching_skills', []))} skills"
        )
    with col3:
        st.metric(
            "Skills to Develop 📈",
            f"{len(match_analysis.get('missing_skills', []))} skills"
        )

    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Skills Analysis 📊",
        "Experience Match 🗂️",
        "Recommendations 💡",
        "Cover Letter 💌",
        "Updated Resume 📝"
    ])

    with tab1:
        st.subheader("Matching Skills")
        for skill in match_analysis.get('matching_skills', []):
            st.success(f"✅ {skill.get('skill_name', 'N/A')}")

        st.subheader("Missing Skills")
        for skill in match_analysis.get('missing_skills', []):
            st.warning(f"⚠️ {skill.get('skill_name', 'N/A')}")
            suggestion = skill.get('suggestion')
            if suggestion:
                st.info(f"Suggestion: {suggestion}")

        matching_skills_count = len(match_analysis.get('matching_skills', []))
        missing_skills_count = len(match_analysis.get('missing_skills', []))
        if matching_skills_count > 0 or missing_skills_count > 0:
            skills_data = pd.DataFrame({
                'Status': ['Matching', 'Missing'],
                'Count': [matching_skills_count, missing_skills_count]
            })
            fig = px.bar(skills_data, x='Status', y='Count', color='Status',
                         color_discrete_sequence=['#5cb85c', '#d9534f'],
                         title='Skills Analysis')
            fig.update_layout(xaxis_title='Status', yaxis_title='Count')
            st.plotly_chart(fig)
        else:
            st.info("No skill data to display.")

    with tab2:
        st.write("### Experience Match Analysis 🗂️")
        st.write(match_analysis.get('experience_match_analysis', 'No analysis provided.'))
        st.write("### Education Match Analysis 🎓")
        st.write(match_analysis.get('education_match_analysis', 'No analysis provided.'))

    with tab3:
        st.write("### Key Recommendations 🔑")
        for rec in match_analysis.get('recommendations_for_improvement', []):
            st.info(f"**{rec.get('recommendation', 'N/A')}**")
            st.write(f"**Section:** {rec.get('section', 'N/A')}")
            st.write(f"**Guidance:** {rec.get('guidance', 'N/A')}")

        st.write("### ATS Optimization Suggestions 🤖")
        for suggestion in match_analysis.get('ats_optimization_suggestions', []):
            st.write("---")
            st.warning(f"**Section to Modify:** {suggestion.get('section', 'N/A')}")
            if suggestion.get('current_content'):
                st.write(f"**Current Content:** {suggestion['current_content']}")
            st.write(f"**Suggested Change:** {suggestion.get('suggested_change', 'N/A')}")
            if suggestion.get('keywords_to_add'):
                st.write(f"**Keywords to Add:** {', '.join(suggestion['keywords_to_add'])}")
            if suggestion.get('formatting_suggestion'):
                st.write(f"**Formatting Changes:** {suggestion['formatting_suggestion']}")
            if suggestion.get('reason'):
                st.info(f"**Reason for Change:** {suggestion['reason']}")

    with tab4:
        st.write("### Cover Letter Generator 🖊️")
        tone = st.selectbox("Select tone 🎭",
                            ["Professional 👔", "Enthusiastic 😃", "Confident 😎", "Friendly 👋"])

        if st.button("Generate Cover Letter ✍️"):
            with st.spinner("✍️ Crafting your cover letter... (Using Pro model)"):
                cover_letter = cover_letter_gen.generate_cover_letter(
                    job_desc_text,
                    resume_text,
                    match_analysis,
                    tone.lower().split()[0]
                )
                st.markdown("### Your Custom Cover Letter 💌")
                st.text_area("", cover_letter, height=400)
                st.download_button(
                    "Download Cover Letter 📥",
                    cover_letter,
                    "cover_letter.txt",
                    "text/plain"
                )

    with tab5:
        st.write("### Updated Resume 📝")
        updated_resume = generate_updated_resume(resume_text, match_analysis)
        st.download_button(
            "Download Updated Resume 📥",
            updated_resume,
            "updated_resume.pdf",
            mime="application/pdf"
        )


# --- State Management Helper ---
def clear_state():
    """Clears all session state variables to restart analysis."""
    keys_to_clear = [
        'match_analysis', 'resume_text', 'job_desc_text',
        'resume_graph', 'job_graph', 'view_graphs'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def main():
    st.set_page_config(page_title="Graph RAG Job Assistant - ResumeSaathi 📝", layout="wide")
    st.title("Graph RAG Job Application Assistant - ResumeSaathi 🚀")

    # API key input
    api_key = st.sidebar.text_input("Enter LLM API Key 🗝️", type="password")
    if not api_key:
        st.warning("🔑 Please enter your LLM API key to continue.")
        st.stop()  # Use st.stop() to halt execution if no key

    # Initialize analyzers
    try:
        job_analyzer = JobAnalyzer(api_key)
        cover_letter_gen = CoverLetterGenerator(api_key)
    except Exception as e:
        st.error(f"Failed to initialize analyzers: {e}")
        st.stop()

    # --- Mode Selection ---
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        (
            "Simple (Fast)",
            "1-to-1 Graph RAG (Deep Analysis)",
            "1-to-Many Batch RAG (Find Best Fit)"
        )
    )

    # --- Add a button to clear state in the sidebar ---
    st.sidebar.button("Start New Analysis", on_click=clear_state)

    # ==================================================================
    # --- MODE 1: Original Simple (Fast) ---
    # ==================================================================
    if analysis_mode == "Original Simple (Fast)":
        st.sidebar.info("Uses the small LLM model for a quick, simple JSON-based analysis of one resume and one job.")

        # --- Use Session State to avoid re-running ---
        if 'match_analysis' in st.session_state:
            # --- If analysis is done, show results ---
            display_1_to_1_ui(
                st.session_state['match_analysis'],
                st.session_state['resume_text'],
                st.session_state['job_desc_text'],
                cover_letter_gen
            )
        else:
            # --- If no analysis, show uploaders ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Job Description 📋")
                job_desc_text = st.text_area("Paste the job description here", height=300, key="simple_jd")

            with col2:
                st.subheader("Your Resume 📜")
                resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx', 'txt'], key="simple_resume")

            if st.button("Run Simple Analysis ⚡") and job_desc_text and resume_file:
                with st.spinner("🔍 Running Simple Analysis... (Using Flash model)"):
                    resume_text = load_text_from_file(resume_file)

                    if resume_text:
                        job_analysis = job_analyzer.analyze_job_simple(job_desc_text)
                        resume_analysis = job_analyzer.analyze_resume_simple(resume_text)

                        if not job_analysis or not resume_analysis:
                            st.error("Failed to analyze job or resume. Cannot proceed.")
                            st.stop()

                        match_analysis = job_analyzer.analyze_match_simple(job_analysis, resume_analysis)

                        if not match_analysis:
                            st.error("Insufficient data returned from match analysis. Please try again.")
                            st.stop()

                        # --- Save to Session State ---
                        st.session_state['match_analysis'] = match_analysis
                        st.session_state['resume_text'] = resume_text
                        st.session_state['job_desc_text'] = job_desc_text

                        # Rerun to display results
                        st.rerun()

    # ==================================================================
    # --- MODE 2: 1-to-1 Graph RAG (Deep Analysis) ---
    # ==================================================================
    elif analysis_mode == "1-to-1 Graph RAG (Deep Analysis)":
        st.sidebar.info(
            "Uses the powerful large LLM model to build detailed knowledge graphs for a deep, contextual analysis of one resume and one job.")

        # --- State Caching Logic ---
        if 'match_analysis' in st.session_state:
            # --- 1. RESULTS EXIST: Show analysis and graph toggle ---
            display_1_to_1_ui(
                st.session_state['match_analysis'],
                st.session_state['resume_text'],
                st.session_state['job_desc_text'],
                cover_letter_gen
            )

            st.markdown("---")
            if st.button("View Knowledge Graphs 📊"):
                st.session_state.view_graphs = True

            if st.session_state.get('view_graphs', False):
                st.header("Generated Knowledge Graphs")
                st.info(
                    "Visualizing the extracted knowledge graphs. Use mouse scroll to zoom and click-and-drag to pan.")

                # --- Define color maps for node types ---
                resume_node_map = {
                    'PERSON': '#FFC107', 'COMPANY': '#EC407A', 'JOBTITLE': '#42A5F5',
                    'SKILL': '#66BB6A', 'DEGREE': '#B39DDB', 'UNIVERSITY': '#B39DDB',
                    'PROJECT': '#FFA726', 'CERTIFICATION': '#FF7043'
                    # --- FIX: Removed 'METRIC' ---
                }
                default_resume_nodes = list(resume_node_map.keys())

                job_node_map = {
                    'ROLE': '#42A5F5', 'COMPANY': '#EC407A', 'REQUIREDSKILL': '#EF5350',
                    'PREFERREDSKILL': '#66BB6A', 'REQUIREDEXPERIENCE': '#AB47BC',
                    'REQUIREDDEGREE': '#B39DDB', 'RESPONSIBILITY': '#78909C'
                }
                default_job_nodes = list(job_node_map.keys())

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.subheader("Resume Graph")
                    # Add filters
                    selected_resume_types = st.multiselect(
                        "Filter Resume Graph Nodes",
                        options=default_resume_nodes,
                        default=default_resume_nodes,
                        key="resume_filter"
                    )

                    try:
                        resume_nodes, resume_edges = convert_json_to_agraph(
                            st.session_state['resume_graph'],
                            resume_node_map,
                            selected_resume_types
                        )
                        # --- FIX: Use container with key for stable rerender ---
                        resume_key = "-".join(selected_resume_types)  # Create a dynamic key
                        with st.container(key=resume_key):
                            agraph(
                                nodes=resume_nodes,
                                edges=resume_edges,
                                config=create_agraph_config()
                                # --- FIX: Removed key from agraph() ---
                            )
                    except Exception as e:
                        st.error(f"Failed to render resume graph: {e}")
                        st.json(st.session_state['resume_graph'])  # Fallback to JSON

                with graph_col2:
                    st.subheader("Job Description Graph")
                    # Add filters
                    selected_job_types = st.multiselect(
                        "Filter Job Graph Nodes",
                        options=default_job_nodes,
                        default=default_job_nodes,
                        key="job_filter"
                    )

                    try:
                        job_nodes, job_edges = convert_json_to_agraph(
                            st.session_state['job_graph'],
                            job_node_map,
                            selected_job_types
                        )
                        # --- FIX: Use container with key for stable rerender ---
                        job_key = "-".join(selected_job_types)  # Create a dynamic key
                        with st.container(key=job_key):
                            agraph(
                                nodes=job_nodes,
                                edges=job_edges,
                                config=create_agraph_config()
                                # --- FIX: Removed key from agraph() ---
                            )
                    except Exception as e:
                        st.error(f"Failed to render job graph: {e}")
                        st.json(st.session_state['job_graph'])  # Fallback to JSON

        else:
            # --- 2. NO RESULTS: Show uploaders to start analysis ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Job Description 📋")
                job_desc_text_input = st.text_area("Paste the job description here", height=300, key="graph_jd")

            with col2:
                st.subheader("Your Resume 📜")
                resume_file_input = st.file_uploader("Upload your resume", type=['pdf', 'docx', 'txt'],
                                                     key="graph_resume")

            if st.button("Run Graph RAG Analysis 🧠") and job_desc_text_input and resume_file_input:
                with st.spinner("🔍 Running Graph RAG Analysis... (This may take a moment, using Pro model)"):
                    resume_text = load_text_from_file(resume_file_input)

                    if resume_text:
                        job_graph = job_analyzer.analyze_job_graph(job_desc_text_input)
                        resume_graph = job_analyzer.analyze_resume_graph(resume_text)

                        if not job_graph or not resume_graph:
                            st.error("Failed to generate one or both graphs. Cannot proceed.")
                            st.stop()

                        match_analysis = job_analyzer.analyze_match_graph(resume_graph, job_graph)

                        if not match_analysis:
                            st.error("Insufficient data returned from graph match analysis. Please try again.")
                            st.stop()

                        # --- Save all results to Session State ---
                        st.session_state['match_analysis'] = match_analysis
                        st.session_state['resume_text'] = resume_text
                        st.session_state['job_desc_text'] = job_desc_text_input
                        st.session_state['resume_graph'] = resume_graph
                        st.session_state['job_graph'] = job_graph

                        # Rerun to display results
                        st.rerun()

    # ==================================================================
    # --- MODE 3: 1-to-Many BATCH ANALYSIS ---
    # ==================================================================
    elif analysis_mode == "1-to-Many Batch RAG (Find Best Fit)":
        st.sidebar.info(
            "Upload one resume and a batch of job descriptions (as files or a .zip) to find your top 3 matches using an efficient best selection pipeline.")

        # This mode is linear, so state isn't as complex
        # We can clear state at the start of this mode if needed
        # For now, we'll keep it simple as it's a one-shot process.

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Resume 📜")
            resume_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'],
                                           key="batch_resume")

        with col2:
            st.subheader("Job Descriptions 📂")
            jd_files = st.file_uploader(
                "Upload all JDs (PDF, DOCX, TXT, or a single .ZIP)",
                type=['pdf', 'docx', 'txt', 'zip'],
                accept_multiple_files=True,
                key="batch_jds"
            )

        if st.button("Find My Top 3 Jobs 🚀") and resume_file and jd_files:
            knowledge_base = []

            with st.spinner("Step 1/5: Processing all job descriptions... (Using Flash model)"):
                all_files = []
                # Handle zip files
                for file in jd_files:
                    if file.name.endswith('.zip'):
                        with zipfile.ZipFile(file, 'r') as z:
                            for filename in z.namelist():
                                if not filename.startswith('__MACOSX') and any(
                                        filename.endswith(ext) for ext in ['.pdf', '.docx', '.txt']):
                                    with z.open(filename) as unzipped_file:
                                        bytes_io = BytesIO(unzipped_file.read())
                                        bytes_io.name = filename
                                        all_files.append(bytes_io)
                    else:
                        all_files.append(file)

                st.info(f"Found {len(all_files)} total job descriptions to process.")

                if not all_files:
                    st.error("No valid files found (PDF, DOCX, TXT). Please check your upload.")
                    st.stop()

                # 1. Process JDs (Lightweight Pass)
                jd_progress = st.progress(0.0)
                for i, file in enumerate(all_files):
                    jd_text = load_text_from_file(file)
                    if jd_text:
                        summary = job_analyzer.summarize_text_for_embedding(jd_text)
                        if not summary:
                            st.warning(f"Could not summarize {file.name}, skipping.")
                            continue
                        embedding = job_analyzer.get_embedding(summary)
                        if embedding:
                            knowledge_base.append({
                                'file_name': file.name,
                                'full_text': jd_text,
                                'embedding': embedding
                            })
                    jd_progress.progress((i + 1) / len(all_files))

                if not knowledge_base:
                    st.error("No valid job descriptions could be processed.")
                    st.stop()

            with st.spinner("Step 2/5: Analyzing your resume... (Using Pro model)"):
                # 2. Process Resume (Heavy Pass)
                resume_text = load_text_from_file(resume_file)
                if not resume_text:
                    st.error("Could not read resume file.")
                    st.stop()
                resume_graph = job_analyzer.analyze_resume_graph(resume_text)
                if not resume_graph:
                    st.error("Failed to generate resume graph. Cannot proceed.")
                    st.stop()

            with st.spinner("Step 3/5: Finding top matches... (Using Flash model)"):
                # 3. Summarize & Embed Resume
                resume_summary = job_analyzer.summarize_graph_for_embedding(resume_graph)
                if not resume_summary:
                    st.error("Failed to summarize resume graph. Cannot proceed.")
                    st.stop()
                resume_embedding = job_analyzer.get_embedding(resume_summary)
                if not resume_embedding:
                    st.error("Failed to generate resume embedding. Cannot proceed.")
                    st.stop()

                # 4. Retrieve
                for item in knowledge_base:
                    item['similarity'] = cosine_similarity(resume_embedding, item['embedding'])

                knowledge_base.sort(key=lambda x: x['similarity'], reverse=True)
                top_3_jobs = knowledge_base[:3]

                st.success(f"Top 3 potential matches found: {', '.join([j['file_name'] for j in top_3_jobs])}")

            with st.spinner("Step 4/5: Performing deep Graph RAG analysis on top 3... (Using Pro model)"):
                # 5. Augment (Graph Generation for Top 3)
                top_3_with_graphs = []
                for job in top_3_jobs:
                    job_graph = job_analyzer.analyze_job_graph(job['full_text'])
                    top_3_with_graphs.append({
                        'file_name': job['file_name'],
                        'graph': job_graph  # This may be {} if it fails, will be filtered later
                    })

            with st.spinner("Step 5/5: Generating final report... (Using Pro model)"):
                # 6. Generate (Final Analysis)
                final_report = job_analyzer.analyze_top_matches(resume_graph, top_3_with_graphs)

            # 7. Display Final Report
            st.header("Batch Analysis Complete: Your Top 3 Matches")

            if final_report:
                best_match = final_report.get('best_match')
                other_matches = final_report.get('other_matches', [])

                if best_match:
                    st.subheader(
                        f"🥇 Best Match: {best_match.get('file_name')} (Score: {best_match.get('match_score')})")
                    st.info(f"**Rationale:** {best_match.get('rationale')}")
                    with st.expander("Show Key Gaps"):
                        for gap in best_match.get('key_gaps', []):
                            st.warning(gap)

                st.subheader("🥈 Other Matches")
                for match in other_matches:
                    st.markdown("---")
                    st.subheader(f"{match.get('file_name')} (Score: {match.get('match_score')})")
                    st.write(f"**Rationale:** {match.get('rationale')}")
                    with st.expander("Show Key Gaps"):
                        for gap in match.get('key_gaps', []):
                            st.warning(gap)
            else:
                st.error("Failed to generate the final analysis report.")


if __name__ == "__main__":
    main()