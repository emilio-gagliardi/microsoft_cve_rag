import os
import yaml
import jinja2
from jinja2 import meta
import textwrap
from datetime import datetime, date
import json
import logging
from application.app_utils import setup_logger

logging.getLogger(__name__)

class CustomLoader(yaml.SafeLoader):
    pass

# Remove the boolean resolver from CustomLoader
for key in list(CustomLoader.yaml_implicit_resolvers):
    resolvers = CustomLoader.yaml_implicit_resolvers[key]
    CustomLoader.yaml_implicit_resolvers[key] = [
        res for res in resolvers if res[0] != 'tag:yaml.org,2002:bool'
    ]

def custom_constructor(loader, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        # Handle explicit booleans ('true' and 'false')
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        # Parse date strings
        try:
            return datetime.fromisoformat(value.rstrip('Z'))
        except ValueError:
            pass
        return value
    elif isinstance(node, yaml.SequenceNode):
        return [custom_constructor(loader, child) for child in node.value]
    elif isinstance(node, yaml.MappingNode):
        return {
            loader.construct_scalar(key): custom_constructor(loader, value)
            for key, value in node.value
        }
    else:
        return loader.construct_object(node)

CustomLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, custom_constructor)


class MongoPipelineLoader:
    """
    Class for loading MongoDB aggregation pipelines from YAML files with Jinja templating.
    """
    def __init__(self, base_directory="etl"):
        """
        Initialize the MongoPipelineLoader.

        :param base_directory: Base directory where YAML files are stored.
        """
        # Set the base directory for YAML files, defaulting to 'etl'
        self.base_directory = base_directory
        self.template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.base_directory))

    def get_pipeline(self, yaml_path, arguments):
        """
        Load and render a MongoDB aggregation pipeline from a YAML file using Jinja templating.

        :param yaml_path: Path to the YAML file, relative to base_directory.
        :param arguments: Dictionary containing arguments to be rendered into the pipeline template
        :return: Rendered pipeline as a list of dictionaries
        :raises ValueError: If unexpected or missing arguments are detected
        """
        try:
            logging.debug(f"Loading pipeline from: {yaml_path}")
            logging.debug(f"Arguments passed to the template: {arguments}")
            # Load the Jinja template
            template = self.template_env.get_template(yaml_path)

            with open(os.path.join(self.base_directory, yaml_path), 'r') as file:
                template_source = file.read()
            parsed_content = self.template_env.parse(template_source)
            valid_variables = meta.find_undeclared_variables(parsed_content)

            # Check for unexpected arguments
            unexpected_arguments = [key for key in arguments if key not in valid_variables]
            if unexpected_arguments:
                raise ValueError(
                    f"Unexpected arguments provided: {unexpected_arguments}. "
                    f"Expected arguments: {valid_variables}."
                )

            # Check for missing arguments
            missing_arguments = [var for var in valid_variables if var not in arguments]
            if missing_arguments:
                raise ValueError(
                    f"Missing arguments for template: {missing_arguments}. "
                    f"Expected arguments: {valid_variables}."
                )

            # Render the Jinja template with the provided arguments
            rendered_yaml = template.render(**arguments)

            # Parse the rendered YAML into a list of dictionaries
            pipeline = yaml.load(rendered_yaml, Loader=CustomLoader)

            return pipeline

        except FileNotFoundError as e:
            logging.error(f"Pipeline file not found: {yaml_path}")
            raise e
        except (yaml.YAMLError, jinja2.TemplateError, ValueError) as e:
            logging.error(f"Error loading or rendering Pipeline file: {yaml_path}: {e}")
            raise e

    def get_base_directory(self):
        """
        Print the base directory for debugging purposes.
        """
        return f"Base directory: {self.base_directory}"

# Example usage
if __name__ == "__main__":
    pass
    # loader = MongoPipelineLoader(base_directory=os.path.join(os.getcwd(), "etl"))

    # # Define arguments to be passed to the template
    # start_date = datetime(2024, 8, 1)
    # end_date = datetime(2024, 8, 2, 23, 59, 59)
    # pipeline_file_path = r"fe_product_build_ids.yaml"
    # exclude_collections = ['archive_stable_channel_notes', 'beta_channel_notes', 'mobile_stable_channel_notes', 'windows_update']
    # # Load the pipeline
    # for yaml_file, arguments in pipelines_arguments_config.items():
    # try:
    #     resolved_pipeline = loader.get_pipeline(yaml_file, arguments)
    #     print(my_pipeline)
    # except Exception as e:
    #     logging.error(f"Failed to load pipeline: {e}")
