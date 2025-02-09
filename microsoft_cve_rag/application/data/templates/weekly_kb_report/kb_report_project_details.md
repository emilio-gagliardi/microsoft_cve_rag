# Project Description: Jinja2 Static Report Template

## Overview

This project involves developing a static report generation system using Python and Jinja2. The generated reports will incorporate modern HTML, CSS, and JavaScript best practices, drawing inspiration from shadcn and Tailwind CSS for functionality and styling—without relying on frameworks such as React or Node.js.

## Objectives

- **Static HTML Generation:** Create a Jinja2 template to produce static HTML files for deployment on a web server.
- **Modern Web Standards:** Ensure the template adheres to the latest standards in web development, including current best practices for HTML, CSS, and JavaScript.
- **Scalability & Maintainability:** Centralize core styling and theme information to maintain consistency and enable easy updates.
- **Responsiveness:** Design the layout so that typography and UI elements adjust fluidly across various screen sizes.

## Data Handling Strategy

- **Preprocessing:** Data extraction and preprocessing will be managed separately in Python.
- **Data Structure:**
  - **KB Articles DataFrame:** Contains the primary records (KB Articles).
  - **CVEs DataFrame:** Serves as a lookup table for CVE information linked to each KB Article. This structure facilitates the ranking and grouping of CVEs based on specific properties.

## UI/UX Components

- **CVE Popup Component:**
  - Each record will include a button that, when clicked, triggers a popup displaying all related CVEs.
  - **Implementation Options:**
    - **Dynamic Data Structure:** Store all CVE data in a JavaScript-accessible data structure that the popup script can reference.
    - **Pre-populated Hidden Divs:** Alternatively, pre-build hidden or off-canvas divs containing the necessary CVE data and use JavaScript to transfer the relevant data into the popup upon activation.
  - The popup should feature smooth opening and closing transitions, employing modern JavaScript techniques.

## Styling & Theming

- **Inspiration:** The visual design and interactive elements should take cues from shadcn and Tailwind CSS, focusing on clean, modular styling without using React or Node.js.
- **Centralized Styles:** Define all core style information in a single location (or file) to act as the theme source, ensuring consistent application of styles across the entire report.
- **Responsiveness:** Utilize responsive design techniques to guarantee that elements stack properly and adjust in size based on the device or viewport dimensions.

## Development Guidelines

1. **Jinja2 Template Construction:**
   - Develop a template that dynamically populates static HTML based on the preprocessed data.
   - Ensure template syntax is clean and modular, facilitating easy updates and maintenance.

2. **Integration of Modern Web Practices:**
   - Write semantic HTML.
   - Leverage CSS for layout and styling, ensuring the separation of concerns.
   - Implement JavaScript for interactive components (e.g., the CVE popup) using unobtrusive and progressive enhancement techniques.

3. **Responsive Design:**
   - Use CSS media queries and flexible layout techniques (e.g., flexbox or grid) to create a fluid, responsive interface.

4. **Mockup Guidance:**
   - Utilize the annotated mockup images as a visual and functional guide.
   - **Important:** The red annotations on the mockups are for guidance only and should not be included in the final rendered template.

5. Folder structure:
If a folder does not exist, create it.
   - **`data`**: Contains the data files for the entire project.
   - **`data/templates/weekly_kb_report`**: Holds the Jinja2 templates for this project.
   - **`data/reports/weekly_kb_report/html`**: Contains the rendered HTML files for the report.
   - **`data/reports/weekly_kb_report/md`**: Contains the rendered Markdown files for the report.
   - **`data/reports/weekly_kb_report/pdf`**: Contains the rendered PDF files for the report.
   - **`data/reports/weekly_kb_report/assets`**: Contains static assets like CSS, JS.
   - **`data/reports/weekly_kb_report/media`**: Contains media files like images and charts.

## Final Notes

This document outlines the core requirements and considerations for developing a modern, responsive static report template using Python, Jinja2, and standard web technologies. The approach balances functionality, scalability, and maintainability while staying true to contemporary best practices.

Feel free to ask if you need further clarification or additional details. This is a solid foundation, and I think it’s a great idea to move forward with this plan. Happy coding!
