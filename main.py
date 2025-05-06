import streamlit as st
import base64
import pandas as pd
from PIL import Image
import io
import os
import time

# Set page configuration
st.set_page_config(
    page_title="VisionFlow: Workflow Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        margin-top: 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .feature-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        text-align: center;
    }
    .metric {
        padding: 1rem;
        background-color: #EFF6FF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .progress-label {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def load_image(image_file):
    img = Image.open(image_file)
    return img


def add_logo():
    # Creating a simple logo - in a real app you would load an actual image
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="display: inline-flex; align-items: center; background-color: #1E3A8A; color: white; padding: 0.5rem 1rem; border-radius: 4px;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">🔍</span>
            <span style="font-weight: 700; font-size: 1.5rem;">VisionFlow</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def generate_sample_output(diagram_type):
    if diagram_type == "Flowchart":
        code = """def binary_search(array, target):
    left = 0
    right = len(array) - 1

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage:
# result = binary_search([1, 2, 3, 4, 5], 3)  # Returns 2"""
        return code

    elif diagram_type == "ER Diagram":
        code = """CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    date_of_birth DATE
);

CREATE TABLE Courses (
    course_id INT PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    credits INT CHECK (credits > 0)
);

CREATE TABLE Enrollments (
    student_id INT,
    course_id INT,
    semester VARCHAR(20),
    grade VARCHAR(2),
    PRIMARY KEY (student_id, course_id, semester),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);"""
        return code


# Main page content
def main():
    # Sidebar
    with st.sidebar:
        add_logo()

        st.markdown("### Upload & Analyze")
        uploaded_file = st.file_uploader("Upload a workflow diagram", type=["jpg", "jpeg", "png"])

        st.markdown("### Settings")
        processing_options = st.multiselect(
            "Processing Options",
            ["Enhanced OCR", "Shape Detection", "AI Code Improvement", "Connection Analysis"],
            ["Shape Detection", "Connection Analysis"]
        )

        output_format = st.radio(
            "Output Format",
            ["Python Code", "SQL Schema", "Auto-detect"]
        )

        if st.button("Process Diagram"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    # Simulate processing time
                    time.sleep(2)
                st.sidebar.success("Processing complete!")
            else:
                st.sidebar.error("Please upload an image first")

        st.markdown("### About")
        st.info("""
        VisionFlow converts workflow diagrams into code using computer vision. 
        Upload a flowchart or ER diagram to generate Python code or SQL schema.
        """)

        st.markdown("### Documentation")
        st.markdown("[View User Guide](#)")
        st.markdown("[GitHub Repository](#)")

    # Main content
    st.markdown('<h1 class="main-header">VisionFlow</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Computer Vision-Based Workflow Analysis System</p>', unsafe_allow_html=True)

    # # Sample images in a row
    # col1, col2, col3 = st.columns([1, 2, 1])
    #
    # with col2:
    #     # Sample image placeholder - in a real app, load your own image
    #     st.image("https://via.placeholder.com/800x400?text=Workflow+Diagram+Example", use_container_width=True)
    #     st.caption("Example: Binary Search Flowchart Analysis")

    # Feature highlights
    st.markdown('<h2 class="feature-header">Analyze Workflow Diagrams with AI</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="feature-header">Flowchart Analysis</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="feature-text">Convert procedural flowcharts into executable Python code with control structures and logic.</p>',
            unsafe_allow_html=True)
        st.code(generate_sample_output("Flowchart"), language="python")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="feature-header">ER Diagram Analysis</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="feature-text">Transform database schema diagrams into SQL DDL statements with tables, relationships, and constraints.</p>',
            unsafe_allow_html=True)
        st.code(generate_sample_output("ER Diagram"), language="sql")
        st.markdown('</div>', unsafe_allow_html=True)

    # System metrics
    st.markdown('<h2 class="feature-header">System Performance</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">92%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Shape Detection Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">87%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Text Extraction Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">25s</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Processing Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">90%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Classification Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Key features section
    st.markdown('<h2 class="feature-header">Key Features</h2>', unsafe_allow_html=True)

    features = [
        {
            "icon": "🔍",
            "title": "Shape Detection",
            "description": "Identifies rectangles, diamonds, circles, and parallelograms with ≥90% accuracy"
        },
        {
            "icon": "📝",
            "title": "Text Extraction",
            "description": "Uses Tesseract OCR to extract text within shapes with ≥85% accuracy"
        },
        {
            "icon": "🔄",
            "title": "Connection Analysis",
            "description": "Maps relationships between elements using Hough Line Transform"
        },
        {
            "icon": "🤖",
            "title": "AI Enhancement",
            "description": "Uses Groq API to improve code generation with error handling"
        },
        {
            "icon": "💻",
            "title": "Code Generation",
            "description": "Produces Python code or SQL schemas from visual workflows"
        },
        {
            "icon": "📊",
            "title": "Workflow Classification",
            "description": "Automatically distinguishes between procedural and database schemas"
        }
    ]

    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #1E3A8A;">{feature['title']}</div>
                <div style="color: #4B5563; font-size: 0.9rem;">{feature['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    # How it works section
    st.markdown('<h2 class="feature-header">How It Works</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        steps = [
            "Image Upload",
            "Preprocessing",
            "Shape Detection",
            "Text Extraction",
            "Connection Analysis",
            "Workflow Classification",
            "Code Generation",
            "AI Enhancement"
        ]

        for i, step in enumerate(steps):
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="background-color: #1E3A8A; color: white; border-radius: 50%; width: 25px; height: 25px; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem;">
                    {i + 1}
                </div>
                <div>{step}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="progress-label">Sample Processing Pipeline</div>', unsafe_allow_html=True)

        # Creating a sample progress visualization
        progress_data = {
            'Step': steps,
            'Time (ms)': [200, 350, 850, 1200, 750, 300, 500, 600]
        }
        df = pd.DataFrame(progress_data)

        # Plot horizontal bar chart
        st.bar_chart(df.set_index('Step'))



    # Call to action
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0; padding: 2rem; background-color: #EFF6FF; border-radius: 8px;">
        <h2 style="color: #1E3A8A; margin-bottom: 1rem;">Ready to Try VisionFlow?</h2>
        <p style="margin-bottom: 1.5rem; color: #4B5563;">Upload your workflow diagram in the sidebar and see the magic happen.</p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>VisionFlow: Computer Vision-Based Workflow Analysis System</p>
        <p>© 2025 MTech Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()