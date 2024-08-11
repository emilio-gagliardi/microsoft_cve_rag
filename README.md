# Microsoft CVE RAG

## Overview

The Microsoft CVE RAG (Common Vulnerabilities and Exposures Risk Assessment Guide) is a comprehensive tool designed to help organizations assess and manage the risk associated with various vulnerabilities. This project leverages multiple databases and services to provide a robust and scalable solution for vulnerability management.

## Features

- **Document Management**: Create, read, update, and delete documents related to vulnerabilities.
- **Graph Database Integration**: Manage and query graph nodes representing various entities and their relationships.
- **Vector Database Integration**: Store and query vector representations of data for advanced analytics.
- **ETL Workflows**: Extract, transform, and load data from various sources.
- **Chat Service**: Interactive chat service for querying and managing data.
- **Embedding Service**: Generate embeddings for text data using pre-trained models.

## Installation

### Prerequisites

- Docker
- Docker Compose
- Python 3.8+
- MongoDB
- Neo4j

### Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/emilio-gagliardi/microsoft_cve_rag.git
    cd microsoft_cve_rag
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up the environment variables:

    ```sh
    cp .env.example .env
    # Update the .env file with your credentials
    ```

5. Run the application:

    ```sh
    docker-compose up --build
    ```

## Usage

### API Endpoints

- **Document Management**:
  - `POST /documents/`: Create a new document.
  - `GET /documents/{document_id}`: Retrieve a document by ID.
  - `PUT /documents/{document_id}`: Update a document by ID.
  - `DELETE /documents/{document_id}`: Delete a document by ID.
  - `POST /documents/query/`: Query documents.

- **Graph Database**:
  - `POST /graph/nodes/`: Create a new graph node.
  - `GET /graph/nodes/{node_id}`: Retrieve a graph node by ID.
  - `PUT /graph/nodes/{node_id}`: Update a graph node by ID.
  - `DELETE /graph/nodes/{node_id}`: Delete a graph node by ID.
  - `POST /graph/query/`: Query the graph database.

- **Vector Database**:
  - `POST /vector/`: Create a new vector.
  - `GET /vector/{vector_id}`: Retrieve a vector by ID.
  - `PUT /vector/{vector_id}`: Update a vector by ID.
  - `DELETE /vector/{vector_id}`: Delete a vector by ID.
  - `POST /vector/query/`: Query the vector database.

### Running Tests

To run the tests, use the following command:

```sh
pytest
```

## Aider -- Sonnet

The `aider --sonnet` command is a powerful tool that helps in generating and managing code changes efficiently. It uses a structured format to ensure that all changes are well-documented and easy to track.

### Usage

To use the `aider --sonnet` command, follow these steps:

1. Open your terminal.
2. Navigate to the root directory of your project.
3. Run the following command:

    ```sh
    aider --sonnet
    ```

This command will guide you through the process of making and documenting code changes.

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact Emilio Gagliardi at [email@example.com](mailto:email@example.com).
