# Software Requirements Specification (SRS) Document for VisionFlow
## 1. Introduction
### 1.1 Purpose
This document outlines the requirements for a Computer Vision-based Workflow Analysis System designed to interpret workflow diagrams from images and generate either a database schema or executable code. The system targets MTech students, researchers, and developers looking to automate the conversion of visual workflows into structured outputs.

### 1.2 Scope
The system will:

Accept images of workflow diagrams as input.
Use computer vision to detect and interpret shapes, text, and connections.
Classify the workflow as either a database schema or procedural logic.
Generate appropriate outputs (SQL schema or code).
Provide a simple interface for input and output review. The project focuses on common workflow notations (e.g., flowcharts, ER diagrams) and supports extensibility for future enhancements.
### 1.3 Definitions, Acronyms, and Abbreviations
CV: Computer Vision
DDL: Data Definition Language (e.g., SQL CREATE statements)
ER Diagram: Entity-Relationship Diagram
OCR: Optical Character Recognition
SRS: Software Requirements Specification
### 1.4 References
OpenCV Documentation (for CV implementation)
Tesseract OCR Documentation (for text extraction)
Research papers on workflow diagram recognition (e.g., from IEEE, ACM)
### 1.5 Overview
This SRS covers functional and non-functional requirements, system architecture, and constraints for an MTech-level project completed within a semester.

## 2. Overall Description
### 2.1 User Needs
Students/Developers: Need a tool to quickly convert hand-drawn or digital workflow diagrams into usable outputs without manual transcription.
Educators: Can use it to evaluate student workflows by generating executable code or schemas.
### 2.2 Assumptions and Dependencies
Input images are clear and follow standard workflow notations (e.g., flowcharts, ER diagrams).
Availability of libraries like OpenCV, Tesseract, and a programming environment (e.g., Python).
Users have basic knowledge of workflows and programming/database concepts.
## 3. Specific Requirements
### 3.1 Functional Requirements
Image Input Module
The system shall accept image files (JPEG, PNG) of workflow diagrams via a user interface.
Maximum file size: 10 MB.
Workflow Detection Module
The system shall use CV techniques to identify shapes (e.g., rectangles for processes/tables, arrows for connections, diamonds for decisions).
The system shall extract text within shapes using OCR.
Workflow Classification Module
The system shall classify the workflow as either a database schema (e.g., ER diagram) or procedural logic (e.g., flowchart).
Classification accuracy target: ≥85%.
Output Generation Module
For database schemas:
Generate SQL DDL statements (e.g., CREATE TABLE with columns and relationships).
Example: For an ER diagram with "Student" and "Course" entities, output includes foreign key constraints.
For procedural workflows:
Generate code in Python (or user-specified language) reflecting the logic.
Example: For a flowchart with a decision, output includes if-else statements.
User Interface
Provide a simple GUI to upload images, display detected workflow elements, and show generated output.
Allow users to edit or confirm the output before saving.
### 3.2 Non-Functional Requirements
Performance
Process an image and generate output within 30 seconds on a standard laptop (e.g., 8GB RAM, i5 processor).
Accuracy
Shape detection accuracy: ≥90%.
Text extraction accuracy: ≥85%.
Usability
Interface shall be intuitive with minimal training required (≤10 minutes).
Scalability
System shall handle workflows with up to 20 elements (shapes) in the initial version.
### 3.3 System Architecture
Input Layer: Image upload and preprocessing (grayscale, noise removal).
CV Layer: Shape detection (OpenCV), text extraction (Tesseract OCR).
Analysis Layer: Rule-based or ML model to classify workflow type and map elements.
Output Layer: Template-based generation of SQL or code.
UI Layer: Tkinter or Flask-based GUI.
### 3.4 External Interface Requirements
Hardware: Webcam (optional) or file system for image input.
Software: Python 3.x, OpenCV, Tesseract, SQLite (for schema testing).
User Interface: Desktop GUI or web-based interface.
## 4. Constraints
Development time: 4–6 months (MTech semester timeline).
Limited to standard workflow notations in the initial version.
No real-time processing; batch image input only.
## 5. Deliverables
Source code with documentation.
User manual.
Test cases and results.
Final report with methodology, results, and future scope.
## 6. Acceptance Criteria
Successfully processes 10 sample workflow images (5 database schemas, 5 procedural flows).
Generates correct output (verified manually) for ≥80% of test cases.
GUI is functional and responsive.
## 7. Future Scope
Support for additional notations (e.g., UML diagrams).
Integration of deep learning for improved accuracy.
Real-time workflow capture via webcam.
